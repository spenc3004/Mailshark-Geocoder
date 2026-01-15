# === Import necessary libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone

from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import io
import os
from dotenv import load_dotenv
import requests
import base64
from openpyxl.chart import BarChart, Reference


# === Helper Functions ===

def feature(category, name):
    return CATEGORY_CONFIG[category]["features"].get(name, False)

def get_category_presets(category):
    return CATEGORY_CONFIG[category]["presets"]

def get_default_weights(category, preset):
    presets = get_category_presets(category)
    w = presets.get(preset, {})
    if w:
        return w
    return CATEGORY_CONFIG[category]["fallback_weights"]

def required_columns_for(category, preset):
    base = CATEGORY_CONFIG[category]["required_columns_base"]
    extra = CATEGORY_CONFIG[category]["required_columns_by_preset"].get(preset, [])
    return list(dict.fromkeys(base + extra))  # preserve order, remove duplicates

def adaptive_minmax_iqr(s: pd.Series, col_name: str, id_series: pd.Series | None = None):
    """
    Adaptive winsorization + min-max to [0,1], WITH AUDIT INFO.
    Returns: (normalized_series, audit_dict)

    audit_dict keys:
      - column (str)
      - has_outliers (bool)
      - lower_fence, upper_fence (float or NaN)
      - num_clipped (int)
      - clipped_row_indices (list[int])
      - clipped_row_ids (list[str])  # uses id_series (e.g., Geocode) if provided
    """
    s = pd.to_numeric(s, errors='coerce')
    s_nonnull = s.dropna()

    audit = {
        "column": col_name,
        "has_outliers": False,
        "lower_fence": np.nan,
        "upper_fence": np.nan,
        "num_clipped": 0,
        "clipped_row_indices": [],
        "clipped_row_ids": [],
        "clipped_original_values": [],
        "clipped_new_values": [],
    }

    if s_nonnull.empty:
        return pd.Series(0.0, index=s.index), audit

    q1, q3 = s_nonnull.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        audit["lower_fence"] = float(lower)
        audit["upper_fence"] = float(upper)

        # detect outliers on the non-null part
        mask_low = s_nonnull < lower
        mask_high = s_nonnull > upper
        has_out = mask_low.any() or mask_high.any()
        audit["has_outliers"] = bool(has_out)

        if has_out:
            # collect full-index positions to report later
            clipped_idx = s.index[(s < lower) | (s > upper)]
            audit["clipped_row_indices"] = clipped_idx.tolist()
            audit["num_clipped"] = int(len(clipped_idx))
            audit["clipped_original_values"] = s.loc[clipped_idx].tolist()
            audit["clipped_new_values"] = s.loc[clipped_idx].clip(lower, upper).tolist()
            if id_series is not None:
                audit["clipped_row_ids"] = id_series.loc[clipped_idx].astype(str).tolist()

            s_clip = s.clip(lower, upper)
        else:
            s_clip = s
    else:
        s_clip = s  # constant column (no spread)

    mn, mx = s_clip.min(), s_clip.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(0.0, index=s.index), audit

    norm = ((s_clip - mn) / (mx - mn)).reindex(s.index).fillna(0.0)
    return norm, audit

def read_uploaded_file(uploaded_file):
    """Load a Streamlit-uploaded CSV or XLSX file into a DataFrame."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")

def coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Coerce selected columns to numeric in-place when present."""
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

def top_quartile_subset(frame: pd.DataFrame, by_col: str) -> pd.DataFrame:
    q = frame[by_col].quantile(0.75)
    return frame[frame[by_col] >= q]

def bracket_revenue_summary(
    frame: pd.DataFrame,
    value_col: str,
    revenue_col: str,
    brackets: list[tuple[int, int | None, str]],
) -> pd.DataFrame:
    values = pd.to_numeric(frame[value_col], errors="coerce")
    revenue = pd.to_numeric(frame[revenue_col], errors="coerce").fillna(0)

    rows = []
    for lower, upper, label in brackets:
        if upper is None:
            mask = values >= lower
        else:
            mask = (values >= lower) & (values <= upper)
        rows.append({"Bracket": label, "Total Revenue": float(revenue[mask].sum())})

    return pd.DataFrame(rows)

def add_revenue_histograms_sheet(
    writer: pd.ExcelWriter,
    working_df: pd.DataFrame,
    revenue_col: str,
):
    income_brackets = [
        (0, 39_999, "0-39,999"),
        (40_000, 74_999, "40,000-74,999"),
        (75_000, 149_999, "75,000-149,999"),
        (150_000, 174_999, "150,000-174,999"),
        (175_000, 249_999, "175,000-249,999"),
        (250_000, None, "250,000+"),
    ]
    home_value_brackets = [
        (0, 149_999, "0-149,999"),
        (150_000, 249_999, "150,000-249,999"),
        (250_000, 399_999, "250,000-399,999"),
        (400_000, 749_999, "400,000-749,999"),
        (750_000, 999_999, "750,000-999,999"),
        (1_000_000, None, "1,000,000+"),
    ]

    sheet_name = "Revenue Histograms"
    income_df = bracket_revenue_summary(working_df, "$ Income", revenue_col, income_brackets)
    home_value_df = bracket_revenue_summary(working_df, "$ Home Value", revenue_col, home_value_brackets)

    income_startrow = 0
    income_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=income_startrow)

    home_value_startrow = income_startrow + len(income_df) + 4
    home_value_df.to_excel(writer, index=False, sheet_name=sheet_name, startrow=home_value_startrow)

    ws = writer.sheets[sheet_name]

    income_chart = BarChart()
    income_chart.type = "col"
    income_chart.title = "Revenue by Income Bracket"
    income_chart.y_axis.title = "Total Revenue"
    income_chart.x_axis.title = "Income Bracket"
    income_data = Reference(
        ws,
        min_col=2,
        min_row=income_startrow + 1,
        max_row=income_startrow + len(income_df) + 1,
    )
    income_cats = Reference(
        ws,
        min_col=1,
        min_row=income_startrow + 2,
        max_row=income_startrow + len(income_df) + 1,
    )
    income_chart.add_data(income_data, titles_from_data=True)
    income_chart.set_categories(income_cats)
    income_chart.height = 12
    income_chart.width = 20
    ws.add_chart(income_chart, "D2")

    home_chart = BarChart()
    home_chart.type = "col"
    home_chart.title = "Revenue by Home Value Bracket"
    home_chart.y_axis.title = "Total Revenue"
    home_chart.x_axis.title = "Home Value Bracket"
    home_data = Reference(
        ws,
        min_col=2,
        min_row=home_value_startrow + 1,
        max_row=home_value_startrow + len(home_value_df) + 1,
    )
    home_cats = Reference(
        ws,
        min_col=1,
        min_row=home_value_startrow + 2,
        max_row=home_value_startrow + len(home_value_df) + 1,
    )
    home_chart.add_data(home_data, titles_from_data=True)
    home_chart.set_categories(home_cats)
    home_chart.height = 12
    home_chart.width = 20
    ws.add_chart(home_chart, f"D{home_value_startrow + 2}")

def write_ranked_workbook(
    writer: pd.ExcelWriter,
    working_df: pd.DataFrame,
    base_summary_dict: dict,
    audit_summary_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    winsor_details_df: pd.DataFrame,
    profile_mode: str | None
):
    # Ranked data
    working_df.to_excel(writer, index=False, sheet_name='Ranked Geocodes')

    # Base summary at the top of 'Summary'
    pd.DataFrame(base_summary_dict).to_excel(writer, sheet_name='Summary', index=False, startrow=0)

    # Winsor summary below it with a blank row
    startrow = (len(base_summary_dict.get("Note", [])) or 1) + 2
    if not audit_summary_df.empty:
        audit_summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=startrow)

    # Profiles (optional)
    if profile_mode == "Dynamic (from file)" and not profiles_df.empty:
        profiles_df.to_excel(writer, sheet_name='Summary', index=True, startrow=startrow + len(audit_summary_df) + 2)

    # Detailed sheet (optional)
    if not winsor_details_df.empty:
        winsor_details_df.to_excel(writer, sheet_name='Winsorized Rows', index=False)

    revenue_col = None
    if "$ Total Spend" in working_df.columns:
        revenue_col = "$ Total Spend"
    elif "Overall Revenue" in working_df.columns:
        revenue_col = "Overall Revenue"
    elif "Revenue" in working_df.columns:
        revenue_col = "Revenue"

    if revenue_col and "$ Income" in working_df.columns and "$ Home Value" in working_df.columns:
        add_revenue_histograms_sheet(writer, working_df, revenue_col)
    else:
        sheet_name = "Revenue Histograms"
        pd.DataFrame(
            {
                "Note": [
                    "Revenue histograms not generated. Missing $ Income, $ Home Value, or revenue column."
                ]
            }
        ).to_excel(writer, sheet_name=sheet_name, index=False)

# === Modeling Helper Functions ===

def smape(y_true, y_pred, eps=1e-8):
    """
    Symmetric MAPE in percent (0..200-ish).
    Handles zeros by adding eps to denominator.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def mean_error(y_true, y_pred):
    """Signed bias: positive => overpredict, negative => underpredict"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred - y_true))

# For champion ranking, bias should be "closer to 0 is better":
def abs_mean_error(y_true, y_pred):
    return abs(mean_error(y_true, y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# sklearn scorers: set greater_is_better=False for "lower is better"
SCORING = {
    "r2": "r2",
    "rmse": make_scorer(rmse, greater_is_better=False),
    "medae": make_scorer(median_absolute_error, greater_is_better=False),
    "smape": make_scorer(smape, greater_is_better=False),
    "me": make_scorer(mean_error, greater_is_better=False),      # report signed (not used for rank)
    "abs_me": make_scorer(abs_mean_error, greater_is_better=False) # used for rank
}

CV = KFold(n_splits=5, shuffle=True, random_state=42)

def crossval_report(model, X, y, needs_scaling: bool):
    """
    Runs CV and returns mean/std for metrics.
    Uses imputer (fill 0) and optional scaler inside each fold.
    """
    steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))

    pipe = Pipeline(steps)

    res = cross_validate(pipe, X, y, scoring=SCORING, cv=CV, n_jobs=-1, return_train_score=False)

    # Remember: for scorers with greater_is_better=False, sklearn returns negative values
    out = {
        "R2_mean": res["test_r2"].mean(),
        "R2_std":  res["test_r2"].std(),

        "RMSE_mean": (-res["test_rmse"]).mean(),
        "RMSE_std":  (-res["test_rmse"]).std(),

        "MedAE_mean": (-res["test_medae"]).mean(),
        "MedAE_std":  (-res["test_medae"]).std(),

        "SMAPE_mean": (-res["test_smape"]).mean(),
        "SMAPE_std":  (-res["test_smape"]).std(),

        # Signed ME (flip sign back)
        "MeanError_mean": (-res["test_me"]).mean(),
        "MeanError_std":  (-res["test_me"]).std(),

        # abs(mean error) for ranking (flip sign back)
        "AbsMeanError_mean": (-res["test_abs_me"]).mean(),
        "AbsMeanError_std":  (-res["test_abs_me"]).std(),
    }
    return out

def make_model_pipeline(model, needs_scaling: bool):
    steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

def build_nn(n_features):
    model = Sequential([
        Dense(32, activation="relu", input_shape=(n_features,)),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# === Load environment variables (e.g., API endpoint for authentication) ===
load_dotenv()
API_SERVER = os.getenv("API_SERVER")

# === Initialize authentication state ===
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False  

# === Render login screen if user is not authenticated ===
if not st.session_state.authenticated:
    st.title("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        # Simple local check remove and uncomment API auth for production use
        if user and pwd:
            st.session_state.authenticated = True
            st.success("Logged in!")
            st.rerun()  # Reload app in authenticated state
            '''try:
                # Encode credentials for API authentication
                credentials = f"{user}:{pwd}"
                encoded_credentials = base64.b64encode(credentials.encode()).decode()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Basic {encoded_credentials}"
                }

                # Validate credentials with external API
                response = requests.get(f"{API_SERVER}/dotnet/auth", headers=headers, timeout=10)

                if response.ok:
                    st.session_state.authenticated = True
                    st.success("Logged in!")
                    st.rerun()  # Reload app in authenticated state
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Authentication failed: {e}")'''
        else:
            st.error("Please enter both username and password.")

# === Main App Logic (after successful login) ===
else:

    CATEGORY_CONFIG = {
        "Home Services": {
            "features": {
                "use_wps": True,
                "use_cpms": True,
                "use_vds": False,
                "profile_modes": ["Dynamic (from file)", "Fixed Standard"]
            },
            "currency_columns": ["$ Income", "$ Home Value", "$ Total Spend", "$ Average Order"],
            "percentage_columns": [],
            "required_columns_base": [
                '$ Income', '$ Home Value', 'Owner Occupied',
                'Median Year Structure Built', 'Distance'
            ],
            "required_columns_by_preset": {
                "Manual": [],
                "Home SRVCS Acquisition (No History)": [],
                "Home SRVCS Acquisition (With History + Suppression)": [
                    'House Count', 'House Penetration%', '$ Total Spend',
                    'Total Visits', '$ Average Order', 'Selected'
                ],
                "Home SRVCS Acquisition (With History, No Suppression)": [
                    'House Count', 'House Penetration%', '$ Total Spend',
                    'Total Visits', '$ Average Order', 'Selected'
                ]
            },
            "presets": {
                "Manual": {},
                "Home SRVCS Acquisition (No History)": {
                    '$ Income': 0.25,
                    '$ Home Value': 0.15,
                    'Owner Occupied': 0.30,
                    'Median Year Structure Built': 0.05,
                    'House Count': 0.00,
                    'House Penetration%': 0.00,
                    '$ Total Spend': 0.00,
                    'Total Visits': 0.00,
                    '$ Average Order': 0.00,
                    'Distance': (-0.25),
                    'Weighted Penetration Score': 0.00,
                    'Customer Profile Match Score': 0.00
                },
                "Home SRVCS Acquisition (With History + Suppression)": {
                    '$ Income': 0.05,
                    '$ Home Value': 0.05,
                    'Owner Occupied': 0.15,
                    'Median Year Structure Built': 0.03,
                    'House Count': 0.10,
                    'House Penetration%': 0.05,
                    '$ Total Spend': 0.07,
                    'Total Visits': 0.03,
                    '$ Average Order': 0.05,
                    'Distance': (-0.07),
                    'Weighted Penetration Score': 0.15,
                    'Customer Profile Match Score': 0.20
                },
                "Home SRVCS Acquisition (With History, No Suppression)": {
                    '$ Income': 0.05,
                    '$ Home Value': 0.05,
                    'Owner Occupied': 0.10,
                    'Median Year Structure Built': 0.02,
                    'House Count': 0.13,
                    'House Penetration%': 0.13,
                    '$ Total Spend': 0.13,
                    'Total Visits': 0.07,
                    '$ Average Order': 0.10,
                    'Distance': (-0.07),
                    'Weighted Penetration Score': 0.05,
                    'Customer Profile Match Score': 0.10
                }
            },
            "fallback_weights": {
                '$ Income': 0.20,
                '$ Home Value': 0.15,
                'Owner Occupied': 0.15,
                'Median Year Structure Built': 0.05,
                'House Count': 0.10,
                'House Penetration%': 0.10,
                '$ Total Spend': 0.10,
                'Total Visits': 0.05,
                '$ Average Order': 0.05,
                'Distance': (-0.05),
                'Weighted Penetration Score': 0.10,
                'Customer Profile Match Score': 0.10
            },
            "profile_defaults": {
                "ideal_income": 65000,
                "ideal_home_value": 180000,
                "ideal_owner": 75,
                "ideal_year_built": 1995,
                "ideal_distance": 25
            }
        },
        "Automotive": {
            "features": {
                "use_wps": False,
                "use_cpms": False,
                "use_vds": True,
                "profile_modes": []
            },
            "currency_columns": ["$ Income", "$ Home Value", "$ Total Spend", "$ Average Order"],
            "percentage_columns": ['5+ Vehicles', '% 4 Vehicles', '% 3 Vehicles', '% 2 Vehicles', '% 1 Vehicle', '% No Vehicle'],
            "required_columns_base": [
                'Distance', '$ Income', '5+ Vehicles', '% 4 Vehicles', '% 3 Vehicles', 
                '% 2 Vehicles', '% 1 Vehicle', '% No Vehicle'
            ],
            "required_columns_by_preset": {
                "Manual": [],
                "Auto Acquisition (No History)": [],
                "Auto Acquisition (With History + No Suppression)": [
                    'House Count', 'House Penetration%', '$ Total Spend',
                    'Total Visits', '$ Average Order', 'Selected'
                ]
            },
            "presets": {
                "Manual": {},
                "Auto Acquisition (No History)": {
                    'Distance': (-0.35),
                    '$ Income': 0.25,
                    'Vehicle Density Score': 0.20,
                },
                "Auto Acquisition (With History + No Suppression)": {
                    'House Count': 0.23,
                    'Distance': (-0.10),
                    '$ Total Spend': 0.17,
                    '$ Average Order': 0.11,
                    '$ Income': 0.07,
                    'House Penetration%': 0.06,
                    'Total Visits': 0.06,
                    'Vehicle Density Score': 0.15,   
                }
            },
            "fallback_weights": {
                'Distance': (-0.35),
                '$ Income': 0.25,
                'Vehicle Density Score': 0.20,
            },  
            "profile_defaults": {}
        }
    }

    # === Main Streamlit App ===

    # --- Streamlit page settings ---
    st.set_page_config(page_title="Mail Shark Geocode Scoring Tool", layout="wide")
    st.title("📬 Mail Shark Geocode Scoring Tool")

    score, roi = st.tabs(["📈 Scoring", "💸 ROI Analysis"])

    with score:

        # --- Select scoring mode from sidebar ---
        st.sidebar.header("Scoring Configuration")
        category_options = ["— Select a category —"] + list(CATEGORY_CONFIG.keys())
        category = st.sidebar.selectbox("Select Category", category_options, index=0)

        preset_choice = None
        if category != "— Select a category —":
            preset_choice = st.sidebar.selectbox("Select Scoring Mode", list(get_category_presets(category).keys()))
            DEFAULT_WEIGHTS = get_default_weights(category, preset_choice)


        # --- Upload penetration report file ---
        uploaded_file = st.file_uploader("Upload your Penetration Report (CSV or XLSX):", type=['csv', 'xlsx'])

        if uploaded_file and category != "— Select a category —":
            # Load the file into DataFrame
            df = read_uploaded_file(uploaded_file)

            # File validation
            required_cols = required_columns_for(category, preset_choice)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns for the selected mode: {', '.join(missing_cols)}")
                st.stop()
                
            st.write("✅ File loaded:", uploaded_file.name)
            st.write("Cleaning the file and forcing numbers...")

            # Convert currency fields from strings to floats
            for col in CATEGORY_CONFIG[category]["currency_columns"]:
                if col in df.columns:
                    df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

            # Convert percentage fields from whole numebers to percents
            if len(CATEGORY_CONFIG[category]["percentage_columns"]) > 0:
                for col in CATEGORY_CONFIG[category]["percentage_columns"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float) / 100.0
            # No percentage columns to process when list is empty.

            # --- Weight sliders ---
            st.sidebar.header("Adjust Weights & Filters")
            valid_weight_keys = dict(DEFAULT_WEIGHTS)

            if not feature(category, "use_wps"):
                valid_weight_keys.pop('Weighted Penetration Score', None)
            if not feature(category, "use_cpms"):
                valid_weight_keys.pop('Customer Profile Match Score', None)
            if not feature(category, "use_vds"):
                valid_weight_keys.pop('Vehicle Density Score', None)

            weights = {}
            for key, default in valid_weight_keys.items():
                has_raw = key in df.columns
                is_allowed_derived =(
                    (key == 'Weighted Penetration Score' and feature(category, "use_wps")) or
                    (key == 'Customer Profile Match Score' and feature(category, "use_cpms")) or
                    (key == 'Vehicle Density Score' and feature(category, "use_vds"))
                )
                if has_raw or is_allowed_derived:
                    weights[key] = st.sidebar.slider(f"{key} Weight", -1.0, 1.0, float(default), 0.01)

            # --- Fail Filter thresholds ---
            st.sidebar.header("Fail Filter Thresholds")
            min_income = st.sidebar.number_input("Min Household Income ($)", value=40000)
            if category == "Automotive" and "$ Income" in df.columns:
                max_income = st.sidebar.number_input("Max Household Income ($)", value=250000)
            else:
                max_income = None
            max_distance = st.sidebar.number_input("Max Distance (miles)", value=50)
            min_owner = st.sidebar.number_input("Min Owner Occupied (%)", value=70) if 'Owner Occupied' in df.columns else None
            max_penetration = st.sidebar.number_input("Max House Penetration (%)", value=11.0) if 'House Penetration%' in df.columns else None


            # --- Customer Profile Mode ---

            profile_modes = CATEGORY_CONFIG[category]["features"].get("profile_modes", [])
            profile_mode = None
            ideal_income = ideal_home_value = ideal_owner = ideal_year_built = ideal_distance = None


            if profile_modes:
                st.sidebar.header("Customer Profile Mode")
                if preset_choice == "Home SRVCS Acquisition (No History)":
                    profile_mode = "Fixed Standard"
                    st.sidebar.info("Profile Mode set to 'Fixed Standard' for this preset.")
                else:
                    profile_mode = st.sidebar.radio("Customer Profile Mode", label_visibility="collapsed", options=["Dynamic (from file)", "Fixed Standard"])

                if profile_mode == "Fixed Standard":
                    st.sidebar.header("Fixed Standard Values")
                    pdft = CATEGORY_CONFIG[category].get("profile_defaults", {}) or {}
                    ideal_income = st.sidebar.number_input("Ideal Income", value=pdft.get("ideal_income", 65000))
                    ideal_distance = st.sidebar.number_input("Ideal Distance (miles for Profile Match)", value=pdft.get("ideal_distance", 25))
            # Only render controls that match columns present in this dataset
                    if "$ Home Value" in df.columns:
                        ideal_home_value = st.sidebar.number_input("Ideal Home Value", value=pdft.get("ideal_home_value", 180000))
                    if "Owner Occupied" in df.columns:
                        ideal_owner = st.sidebar.number_input("Ideal Owner Occupied %", value=pdft.get("ideal_owner", 75))
                    if "Median Year Structure Built" in df.columns:
                        ideal_year_built = st.sidebar.number_input("Ideal Median Year Built", value=pdft.get("ideal_year_built", 1995))
                elif profile_mode == "Dynamic (from file)":
                    # Dynamic profile based on existing customers in file
                    st.sidebar.header("Dynamic Profile (from file)")
                    driver_candidates = ['House Count', 'House Penetration%', '$ Total Spend', 'Total Visits', '$ Average Order']
                    available_drivers = [c for c in driver_candidates if c in df.columns]
                    if not available_drivers:
                        st.sidebar.warning("⚠️ No valid driver columns found in file for Dynamic profile calculation.")
                    else:
                        driver = st.sidebar.selectbox("Select Driver for Ideal Profile Calculation", options=available_drivers, index=0)
                        st.sidebar.info(f"Ideal values computed from top 25% {driver} in the file.")



                        ideal_map = {
                            "$ Income": "Ideal Income",
                            "$ Home Value": "Ideal Home Value",
                            "Owner Occupied": "Ideal Owner Occupied",
                            "Median Year Structure Built": "Ideal Median Year Built",
                            "Distance": "Ideal Distance",
                        }
                        profiles = {}
                        for drv in available_drivers:
                            top = top_quartile_subset(df, by_col=drv)
                            profiles[drv] = {
                                row_name: (top[col].mean() if col in top.columns else None)
                                for col, row_name in ideal_map.items()
                            }

                        profiles_df = pd.DataFrame(profiles)
                        profiles_df.index.name = "Ideal"

                        ideal_income        = profiles_df.loc["Ideal Income", driver]                     if "Ideal Income" in profiles_df.index else None
                        ideal_home_value    = profiles_df.loc["Ideal Home Value", driver]                 if "Ideal Home Value" in profiles_df.index else None
                        ideal_owner         = profiles_df.loc["Ideal Owner Occupied", driver]             if "Ideal Owner Occupied" in profiles_df.index else None
                        ideal_year_built    = profiles_df.loc["Ideal Median Year Built", driver]          if "Ideal Median Year Built" in profiles_df.index else None
                        ideal_distance      = profiles_df.loc["Ideal Distance", driver]                   if "Ideal Distance" in profiles_df.index else None

            # --- Score generation trigger ---
            if st.button("🚀 Generate Scores & Report"):
                working = df.copy()
                winsor_audits = []  # collect per-column audit dicts
                id_series = working['Geocode'] if 'Geocode' in working.columns else None
                
                # Normalize predictors we actually care about:
                # union of (valid weights keys) and some known numeric fields like Distance
                predictors_to_normalize = set()
                predictors_to_normalize |= set([k for k in valid_weight_keys.keys() if k in working.columns])
                predictors_to_normalize.add("Distance")  

                for col in predictors_to_normalize:
                    try:
                        norm, audit = adaptive_minmax_iqr(working[col], col_name=col, id_series=id_series)
                        working[f"{col}_Norm"] = norm
                        winsor_audits.append(audit)
                    except Exception:
                        # skip non-numeric or any failure silently
                        pass


                # Calculate Weighted Penetration Score with adaptive winsorization
                # Feature gated, if disabled do not create or calculate WPS columns at all
                if feature(category, "use_wps"):
                    if all(col in working.columns for col in ['$ Total Spend', 'Total Visits', 'Selected']):
                        wps = (working['$ Total Spend'] + working['Total Visits']) / working['Selected'].replace(0, np.nan)
                        wps = wps.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                        working['Weighted Penetration Score'] = wps
                        wps_norm, wps_audit = adaptive_minmax_iqr(wps, col_name='Weighted Penetration Score', id_series=id_series)
                        winsor_audits.append(wps_audit)
                        working['Weighted Penetration Score_Norm'] = wps_norm
                    else:
                        st.warning("⚠️ Missing data for Weighted Penetration Score. Skipping this metric.")
                        working['Weighted Penetration Score_Norm'] = 0.0

                # Calculate Customer Profile Match Score
                if feature(category, "use_cpms") and profile_modes:
                    pairs = []

                    if "$ Income" in working.columns and ideal_income is not None:
                        pairs.append(("$ Income", ideal_income))
                    if "$ Home Value" in working.columns and ideal_home_value is not None:
                        pairs.append(("$ Home Value", ideal_home_value))
                    if "Median Year Structure Built" in working.columns and ideal_year_built is not None:
                        pairs.append(("Median Year Structure Built", ideal_year_built))
                    if "Distance" in working.columns and ideal_distance is not None:
                        pairs.append(("Distance", ideal_distance))

                    diffs = []
                    denom_ideals = []

                    for field, ideal in pairs:
                        if field in working.columns:
                            if field == 'Distance':
                                diff = np.where(working[field] <= ideal, 0, working[field] - ideal)
                            else:
                                diff = np.where(working[field] >= ideal, 0, ideal - working[field])
                            diffs.append(diff)
                            denom_ideals.append(ideal)
                    if diffs and denom_ideals:
                        match_score = 1 - (np.sum(diffs, axis=0) / np.sum(denom_ideals))
                        match_score = np.clip(match_score, 0, 1)
                        working['Customer Profile Match Score'] = match_score

                # Calculate Vehicle Density Score
                if feature(category, "use_vds"):
                    VEHICLE_COLUMNS = {
                        '5+ Vehicles': 5,
                        '% 4 Vehicles': 4,
                        '% 3 Vehicles': 3,
                        '% 2 Vehicles': 2,
                        '% 1 Vehicle': 1,
                        '% No Vehicle': -5
                    }

                    # Ensure missing vehicle columns do not crash the app
                    for col in VEHICLE_COLUMNS.keys():
                        if col not in working.columns:
                            working[col] = 0

                    # Calculate raw Vehicle Density Score
                    vds = np.zeros(len(working))
                    for col, points in VEHICLE_COLUMNS.items():
                        vds += working[col] * points
                        working['Vehicle Density Score'] = vds

                                      
                    vds_norm, vds_audit = adaptive_minmax_iqr(vds, col_name='Vehicle Density Score', id_series=id_series)
                    winsor_audits.append(vds_audit)
                    working['Vehicle Density Score_Norm'] = vds_norm

                # Composite scoring calculation
                score = np.zeros(len(working))
                for key, w in weights.items():
                    if w == 0:
                        continue
                    if key == 'Customer Profile Match Score' and key in working.columns:
                        score += working[key] * w
                    elif key == 'Weighted Penetration Score' and 'Weighted Penetration Score_Norm' in working.columns:
                        score += working['Weighted Penetration Score_Norm'] * w
                    elif key == 'Vehicle Density Score' and 'Vehicle Density Score_Norm' in working.columns:
                        score += working['Vehicle Density Score_Norm'] * w
                    elif f"{key}_Norm" in working.columns:
                        score += working[f"{key}_Norm"] * w

                working['Composite Score'] = score

                # Flagging high penetration & filtering
                if max_penetration is not None and 'House Penetration%' in working.columns:
                    working['Penetration Flag'] = np.where(
                        working['House Penetration%'] > max_penetration,
                        "⚠️ Above Max Penetration", ""
                    )
                working['Status'] = np.where(
                    ((max_income is None) | (working['$ Income'] <= max_income)) &
                    (working['$ Income'] >= min_income) &
                    (working['Distance'] <= max_distance) &
                    (working['Owner Occupied'] >= min_owner),
                    "Included", "Excluded"
                )

                # Sort by included status and composite score
                working = working.sort_values(by=['Status', 'Composite Score'], ascending=[False, False])

                # Display histogram and data table
                fig = px.histogram(working, x='Composite Score', title="Composite Score Distribution")
                st.plotly_chart(fig)
                st.dataframe(working)

                # === Export results as Excel ===

                # Summary per column
                audit_summary_df = pd.DataFrame([{
                    "Column": a["column"],
                    "Outliers Detected": a["has_outliers"],
                    "Lower Fence": a["lower_fence"],
                    "Upper Fence": a["upper_fence"],
                    "Rows Winsorized (count)": a["num_clipped"],
                    "Winsorized Geocodes (preview)": ", ".join(a["clipped_row_ids"][:10]) if a["clipped_row_ids"] else ""
                } for a in winsor_audits])

                # Detailed listing: one row per (column, geocode) that was clipped
                detail_rows = []
                for a in winsor_audits:
                    if a.get("has_outliers") and a.get("clipped_row_indices"):
                        for rid, orig, new in zip(
                            a.get("clipped_row_ids", []),
                            a.get("clipped_original_values", []),
                            a.get("clipped_new_values", [])
                        ):
                            detail_rows.append({
                                "Column": a.get("column"),
                                "Geocode": rid,
                                "Row Index": a.get("clipped_row_indices", [])[a.get("clipped_row_ids", []).index(rid)] if a.get("clipped_row_ids", []) else None,
                                "Original Value": orig,
                                "Clipped Value": new
                            })
                winsor_details_df = pd.DataFrame(detail_rows)

                # Base summary 
                base_summary = {
                    "Note": [],
                    "Value": [],
                }
                if profile_mode:
                    base_summary["Note"].append("Profile Mode Used:")
                    base_summary["Value"].append(profile_mode)
                
                if ideal_income is not None:
                    base_summary["Note"].append("Ideal Income:")
                    base_summary["Value"].append(ideal_income)
                if ideal_home_value is not None:
                    base_summary["Note"].append("Ideal Home Value:")
                    base_summary["Value"].append(ideal_home_value)
                if ideal_owner is not None:
                    base_summary["Note"].append("Ideal Owner Occupied %:")
                    base_summary["Value"].append(ideal_owner)
                if ideal_year_built is not None:
                    base_summary["Note"].append("Ideal Year Built:")
                    base_summary["Value"].append(ideal_year_built)
                if ideal_distance is not None:
                    base_summary["Note"].append("Ideal Distance:")
                    base_summary["Value"].append(ideal_distance)

                profiles_df = locals().get("profiles_df", pd.DataFrame())

                # Keep everything we need for later ROI export
                st.session_state["ranked_parts"] = {
                    "working_df": working.copy(),
                    "base_summary_dict": base_summary.copy(),
                    "audit_summary_df": audit_summary_df.copy(),
                    "profiles_df": profiles_df.copy(),
                    "winsor_details_df": winsor_details_df.copy(),
                    "profile_mode": profile_mode,
                }


                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                    write_ranked_workbook(
                        writer=writer,
                        working_df=st.session_state["ranked_parts"]["working_df"],
                        base_summary_dict=st.session_state["ranked_parts"]["base_summary_dict"],
                        audit_summary_df=st.session_state["ranked_parts"]["audit_summary_df"],
                        profiles_df=st.session_state["ranked_parts"]["profiles_df"],
                        winsor_details_df=st.session_state["ranked_parts"]["winsor_details_df"],
                        profile_mode=st.session_state["ranked_parts"]["profile_mode"],
                    )

                st.session_state["ranked_xlsx_bytes"] = towrite.getvalue()


                st.download_button(
                    label="📥 Download Ranked Excel",
                    data=st.session_state["ranked_xlsx_bytes"],
                    file_name=f"{uploaded_file.name.split('.')[0]}_Ranked.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


    with roi:

        parts = st.session_state.get("ranked_parts")
        if not parts:
            st.warning("Run '🚀 Generate Scores & Report' first to populate ranked data.")
            st.stop()

        ranked_df = parts["working_df"]

        roi_file = st.file_uploader("Upload your ROI Export Report (CSV or XLSX):", type=['csv', 'xlsx'])

        

        if roi_file:
            st.write("✅ File loaded:", roi_file.name)
            roi_df = read_uploaded_file(roi_file)

            # Ensure numeric for safe summations
            coerce_numeric_columns(
                roi_df,
                ["Mailing Qty", "RO's", "Responded", "Revenue", "Expense", "Response"],
            )

            # 1) Aggregate by Campaigns (one row per Campaign/Geocode)
            if "Campaigns" not in roi_df.columns:
                st.error("⚠️ 'Campaigns' column not found in ROI file.")
                st.stop()
            else:
                roi_summary = (
                    roi_df
                    .groupby("Campaigns", as_index=False)
                    .agg(**{
                        "Times Mailed To": ("Campaigns", "size"), 
                        "Overall Mailing Qty": ("Mailing Qty", "sum"),
                        "Overall RO's": ("RO's", "sum"),
                        "Overall Responded": ("Responded", "sum"),
                        "Overall Revenue": ("Revenue", "sum"),
                        "Overall Expense": ("Expense", "sum"),
                    })
                )



            # 2) Total Response Rate = total responded / total mailed
            roi_summary["Response Rate"] = np.where(
                roi_summary["Overall Mailing Qty"] > 0,
                roi_summary["Overall Responded"] / roi_summary["Overall Mailing Qty"],
                np.nan
            )


            roi_summary["Overall ROAS"] = np.where(
                (roi_summary["Overall Mailing Qty"] > 0) & (roi_summary["Overall Expense"] > 0),
                roi_summary["Overall Revenue"] / roi_summary["Overall Expense"],
                np.nan
            )

            # 3) Merge the summary to ranked on Geocode (left) vs Campaigns (right)
            merged = ranked_df.merge(roi_summary, left_on="Geocode", right_on="Campaigns", how="left")

            if st.button("Analyze"):
                data = merged.copy()

                # 4) Hide the right-side key after merge
                if "Campaigns" in data.columns:
                    data = data.drop(columns=["Campaigns"])

                # Percentage
                if "Response Rate" in data.columns:
                    data["Response Rate %"] = (data["Response Rate"] * 100).round(2)
                st.success("✅ ROI aggregated and merged successfully!")

                data['Route_Activity'] = np.where(data['Times Mailed To'] >= 3, 'Active', 'Inactive')

                active = data[data['Route_Activity'] == 'Active'].copy()
                inactive = data[data['Route_Activity'] == 'Inactive'].copy()

                candidate_predictors = [
                    'Composite Score',
                    '$ Average Order_Norm',
                    'Total Visits_Norm',
                    '$ Income_Norm',
                    '$ Total Spend_Norm',
                    'House Penetration%_Norm',
                    'Owner Occupied_Norm',
                    'Median Year Structure Built_Norm',
                    'Distance_Norm',
                    '$ Home Value_Norm',
                    'House Count_Norm',
                    'Weighted Penetration Score_Norm',
                    'Customer Profile Match Score',
                    
                ]

                predictors = [c for c in candidate_predictors if c in data.columns]
        
                target = 'Overall ROAS'

                # === Prepare Training Data ===

                X = active[predictors].copy()

                # Add missing indicators for training data
                for col in X.columns:
                    if X[col].isna().any():
                        X[f"{col}_was_missing"] = X[col].isna().astype(int)

                # Fill NaNs with 0
                X_filled = X.fillna(0)

                y = active[target]

                # === Define Models ===
                # Optional: keep a holdout split ONLY for your plots (not for champion selection)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_filled, y, test_size=0.2, random_state=42
                )

                # Build the NN wrapper (don’t predict yet — it must be fit by sklearn/pipeline)
                nn = KerasRegressor(
                    model=build_nn,
                    model__n_features=X_filled.shape[1],
                    epochs=100,
                    batch_size=16,
                    verbose=0,
                    random_state=42
                )


                models = {
                    "Linear Regression": (LinearRegression(), True),
                    "Decision Tree": (DecisionTreeRegressor(random_state=42), False),
                    "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=42), False),
                    "Gradient Boosting": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42), False),
                    "Neural Network": (nn, True),
                }

                # === Model Evaluation ===

                cv_rows = {}
                for name, (model, needs_scaling) in models.items():
                    cv_rows[name] = crossval_report(model, X_filled, y, needs_scaling=needs_scaling)

                cv_df = pd.DataFrame(cv_rows).T

                # -----------------------
                # Champion selection:
                # 1) Rank by MEAN performance (AvgRank)
                # 2) Tie-break by STABILITY (AvgStd: lower std is better)
                # -----------------------
                rank_cols = {
                    "R2_mean": False,                
                    "RMSE_mean": True,                
                    "MedAE_mean": True,
                    "SMAPE_mean": True,
                    "AbsMeanError_mean": True,
                }

                # ranks based on MEANS
                ranks = pd.DataFrame(index=cv_df.index)
                for col, asc in rank_cols.items():
                    ranks[col + "_rank"] = cv_df[col].rank(ascending=asc, method="average")

                cv_df["AvgRank"] = ranks.mean(axis=1)

                # STABILITY measures
                # (R2_std isn't super useful here; we focus on error metrics)
                std_cols = ["RMSE_std", "MedAE_std", "SMAPE_std", "AbsMeanError_std"]
                cv_df["AvgStd"] = cv_df[std_cols].mean(axis=1)

                # sort by AvgRank first, then AvgStd (tie-break)
                cv_df = cv_df.sort_values(["AvgRank", "AvgStd"])

                champion_name = cv_df.index[0]

                st.subheader("Cross-Validated Model Comparison (5-fold)")
                st.write(f"Champion (lowest AvgRank): **{champion_name}**")
                st.dataframe(cv_df)
                st.caption("AvgRank = mean-based ranking across metrics. AvgStd = tie-breaker (lower = more stable).")


                # -----------------------
                # Fit champion pipeline on ALL active data
                # -----------------------
                champ_model, champ_needs_scaling = models[champion_name]

                champion_pipe = make_model_pipeline(clone(champ_model), champ_needs_scaling)
                champion_pipe.fit(X_filled, y)

                # === Residuals on FULL ACTIVE DATA (actuals) ===

                # X_filled and y (from above) already represent all active rows
                X_active_full = X_filled
                y_active_full = y

                # Champion predictions on FULL active data
                active_preds = {}
                active_preds[champion_name] = champion_pipe.predict(X_active_full)

                residuals_df = active[['Overall ROAS']].copy()
                residuals_df['Geocode'] = active['Geocode']

                for name, preds in active_preds.items():
                    residuals_df[f'{name}_residual'] = y_active_full - preds

                
                # === Flag Over / Under performers for each model ===

                for name in active_preds.keys():
                    col = f'{name}_residual'
                    flag_col = f'{name}_performance'

                    residuals_df[flag_col] = np.where(
                        residuals_df[col] > 0,
                        'Over-performer',      # actual > predicted
                        np.where(
                            residuals_df[col] < 0,
                            'Under-performer',  # actual < predicted
                            'On target'
                        )
                    )

                # === Champion model over / under-performers ===

                champ_residual_col = f'{champion_name}_residual'
                champ_flag_col = f'{champion_name}_performance'

                # Sort to find biggest over- and under-performers
                top_over = residuals_df.sort_values(champ_residual_col, ascending=False).head(20)
                top_under = residuals_df.sort_values(champ_residual_col, ascending=True).head(20)

                st.subheader(f"Champion Model Over/Under Performers ({champion_name})")

                st.write("**Top Over-performers (Actual ROAS > Predicted ROAS):**")
                st.dataframe(top_over)

                st.write("**Top Under-performers (Actual ROAS < Predicted ROAS):**")
                st.dataframe(top_under)

                # === Bar chart of champion residuals (using Geocode) ===
                st.subheader(f"{champion_name} Residuals by Route (Active Data)")

                # Keep only what we need for the plot
                residuals_plot = residuals_df[['Geocode', champ_residual_col]].copy()

                # Sort by residual so the bar chart is ordered
                residuals_plot = residuals_plot.sort_values(champ_residual_col)

                # Limit to 50 points for readability
                residuals_plot = residuals_plot.head(50)

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(
                    data=residuals_plot,
                    x='Geocode',                          
                    y=champ_residual_col,
                    ax=ax
                )
                ax.axhline(0, linestyle='--', color='red')
                ax.set_xlabel("Geocode")
                ax.set_ylabel("Residual (Actual - Predicted ROAS)")
                ax.set_title(
                    f"{champion_name} Residuals on Active Data\n"
                    "(Above 0 = Over-performer, Below 0 = Under-performer)"
                )
                plt.xticks(rotation=90)
                st.pyplot(fig)



                # === Prepare Inactive Data for Prediction ===

                X_inactive = inactive[predictors].copy()

                # Handle missing indicators for inactive data
                for col in predictors:
                    if f"{col}_was_missing" in X_train.columns:
                        if X_inactive[col].isna().any():
                            st.warning(f"Column '{col}' has {X_inactive[col].isna().sum()} missing values. Filling with 0 and adding indicator column.")
                        X_inactive[f"{col}_was_missing"] = X_inactive[col].isna().astype(int)

                # Fill NaNs with 0
                X_inactive_filled = X_inactive.fillna(0)


                inactive['Predicted_ROAS'] = champion_pipe.predict(X_inactive_filled)


                # === Build unified ranked routes table (active + inactive) ===

                # Champion predictions for ACTIVE routes 
                active_export = active.copy()
                active_export['Route_Activity'] = 'Active'
                active_export['Actual_ROAS'] = active[target]

                # champion predictions on active data
                active_export['Predicted_ROAS'] = active_preds[champion_name]

                # residual + performance flag from residuals_df
                active_export['Residual'] = residuals_df[champ_residual_col]
                active_export['Performance_Flag'] = residuals_df[champ_flag_col]

                # Keep only useful columns for export
                active_export = active_export[[
                    'Geocode',
                    'Route_Activity',
                    'Actual_ROAS',
                    'Predicted_ROAS',
                    'Residual',
                    'Performance_Flag'
                ]]

                # INACTIVE routes export
                inactive_export = inactive.copy()
                inactive_export['Route_Activity'] = 'Inactive'
                inactive_export['Actual_ROAS'] = np.nan
                inactive_export['Residual'] = np.nan
                inactive_export['Performance_Flag'] = 'Inactive'

                inactive_export = inactive_export[[
                    'Geocode',
                    'Route_Activity',
                    'Actual_ROAS',
                    'Predicted_ROAS',
                    'Residual',
                    'Performance_Flag'
                ]]

                # Combine active + inactive
                ranked_routes = pd.concat([active_export, inactive_export], ignore_index=True)

                # Rank by predicted ROAS (highest first)
                ranked_routes = ranked_routes.sort_values('Predicted_ROAS', ascending=False)
                ranked_routes['Predicted_ROAS_Rank'] = ranked_routes['Predicted_ROAS'].rank(
                    ascending=False,
                    method='dense'
                )

                st.subheader("Ranked Routes (Active + Inactive)")
                st.dataframe(ranked_routes.head(50))  

                # === Build underperformer -> recommended route mapping (Sheet 2) ===

                # Underperforming ACTIVE routes (most negative residuals first)
                underperformers = active_export[active_export['Performance_Flag'] == 'Under-performer'] \
                    .sort_values('Residual')  # most negative at top

                # Recommended routes: INACTIVE with highest predicted ROAS
                recommended_routes = inactive_export.sort_values('Predicted_ROAS', ascending=False)

                # Pair them up 1-to-1 (min length of the two lists)
                n_pairs = min(len(underperformers), len(recommended_routes))

                mapping_df = pd.DataFrame({
                    'Underperforming_Route_Geocode': underperformers['Geocode'].head(n_pairs).values,
                    'Underperforming_Residual': underperformers['Residual'].head(n_pairs).values,
                    'Recommended_Route_Geocode': recommended_routes['Geocode'].head(n_pairs).values,
                    'Recommended_Predicted_ROAS': recommended_routes['Predicted_ROAS'].head(n_pairs).values
                })

                st.subheader("Underperformer → Recommended Route Mapping (Preview)")
                st.dataframe(mapping_df.head(20))



                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as w:
                    write_ranked_workbook(
                        writer=w,
                        working_df=parts["working_df"],
                        base_summary_dict=parts["base_summary_dict"],
                        audit_summary_df=parts["audit_summary_df"],
                        profiles_df=parts["profiles_df"],
                        winsor_details_df=parts["winsor_details_df"],
                        profile_mode=parts["profile_mode"],
                )
                    data.to_excel(w, index=False, sheet_name='Ranked + ROI Summary')
                    roi_summary.to_excel(w, index=False, sheet_name='ROI Summary by Campaign')
                    ranked_routes.to_excel(w, index=False, sheet_name='Ranked Routes with Predictions')
                    mapping_df.to_excel(w, index=False, sheet_name='Route Mapping')

                out.seek(0)

                st.download_button(
                    "📥 Download Ranked + ROI Summary",
                    data=out.getvalue(),
                    file_name="ranked_with_roi_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # === Visualization ===
                st.subheader("Champion Model Visualizations")

                y_pred = champion_pipe.predict(X_val)  
                residuals = y_val - y_pred

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                sns.scatterplot(x=y_val, y=y_pred, ax=axes[0])
                axes[0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
                axes[0].set_xlabel("Actual ROAS")
                axes[0].set_ylabel("Predicted ROAS")
                axes[0].set_title(f"{champion_name}: Actual vs Predicted")

                sns.scatterplot(x=y_pred, y=residuals, ax=axes[1])
                axes[1].axhline(0, color='r', linestyle='--')
                axes[1].set_xlabel("Predicted ROAS")
                axes[1].set_ylabel("Residuals")
                axes[1].set_title(f"{champion_name}: Residuals")

                st.pyplot(fig)
