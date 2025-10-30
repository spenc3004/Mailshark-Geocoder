# === Import necessary libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from dotenv import load_dotenv
import requests
import base64


# === Helper Functions ===

# === Adaptive Winsorization Function (with audit) ===
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


def top_quartile_subset(frame: pd.DataFrame, by_col: str) -> pd.DataFrame:
    q = frame[by_col].quantile(0.75)
    return frame[frame[by_col] >= q]

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
                "profile_modes": ["Dynamic (from file)", "Fixed Standard"]
            },
            "currency_columns": ["$ Income", "$ Home Value", "$ Total Spend", "$ Average Order"],
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
                "profile_modes": []
            },
            "currency_columns": ["$ Income", "$ Home Value", "$ Total Spend", "$ Average Order"],
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
                    '5+ Vehicles': 0.20,
                    '% 4 Vehicles': 0.12,
                    '% 3 Vehicles': 0.07,
                    '% 2 Vehicles': 0.05,
                    '% 1 Vehicle': 0.03,
                    '% No Vehicle': (-0.07)
                },
                "Auto Acquisition (With History + No Suppression)": {
                    'House Count': 0.020,
                    'Distance': (-0.15),
                    '$ Total Spend': 0.12,
                    '$ Average Order': 0.10,
                    '$ Income': 0.07,
                    'House Penetration%': 0.05,
                    'Total Visits': 0.05,
                    '5+ Vehicles': 0.15,
                    '% 4 Vehicles': 0.07,
                    '% 3 Vehicles': 0.05,
                    '% 2 Vehicles': 0.04,
                    '% 1 Vehicle': 0.02,
                    '% No Vehicle': (-0.07)    
                }
            },
            "fallback_weights": {
                'Distance': (-0.35),
                '$ Income': 0.25,
                '5+ Vehicles': 0.20,
                '% 4 Vehicles': 0.12,
                '% 3 Vehicles': 0.07,
                '% 2 Vehicles': 0.05,
                '% 1 Vehicle': 0.03,
                '% No Vehicle': (-0.07)
            },  
            "profile_defaults": {}
        }
    }

    # === Helper functions ===

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
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.lower().endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

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

            # --- Weight sliders ---
            st.sidebar.header("Adjust Weights & Filters")
            valid_weight_keys = dict(DEFAULT_WEIGHTS)

            if not feature(category, "use_wps"):
                valid_weight_keys.pop('Weighted Penetration Score', None)
            if not feature(category, "use_cpms"):
                valid_weight_keys.pop('Customer Profile Match Score', None)

            weights = {}
            for key, default in valid_weight_keys.items():
                has_raw = key in df.columns
                is_allowed_derived =(
                    (key == 'Weighted Penetration Score' and feature(category, "use_wps")) or
                    (key == 'Customer Profile Match Score' and feature(category, "use_cpms"))
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

                # Composite scoring calculation
                score = np.zeros(len(working))
                for key, w in weights.items():
                    if w == 0:
                        continue
                    if key == 'Customer Profile Match Score' and key in working.columns:
                        score += working[key] * w
                    elif key == 'Weighted Penetration Score' and 'Weighted Penetration Score_Norm' in working.columns:
                        score += working['Weighted Penetration Score_Norm'] * w
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
            if roi_file.name.lower().endswith('.csv'):
                roi_df = pd.read_csv(roi_file)
            elif roi_file.name.lower().endswith('.xlsx'):
                roi_df = pd.read_excel(roi_file)
            


            # Ensure numeric for safe summations
            for c in ["Mailing Qty", "RO's", "Responded", "Revenue", "Expense", "Response"]:
                if c in roi_df.columns:
                    roi_df[c] = pd.to_numeric(roi_df[c], errors="coerce")

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
                analysis = merged.copy()

                # 4) Hide the right-side key after merge
                if "Campaigns" in analysis.columns:
                    analysis = analysis.drop(columns=["Campaigns"])

                # Percentage
                if "Response Rate" in analysis.columns:
                    analysis["Response Rate %"] = (analysis["Response Rate"] * 100).round(2)

                st.success("✅ ROI aggregated and merged successfully!")
                st.dataframe(analysis)


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
                    analysis.to_excel(w, index=False, sheet_name='Ranked + ROI Summary')
                    roi_summary.to_excel(w, index=False, sheet_name='ROI Summary by Campaign')


                st.download_button(
                    "📥 Download Ranked + ROI Summary",
                    data=out.getvalue(),
                    file_name="ranked_with_roi_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
