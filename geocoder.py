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
        if user and pwd:
            try:
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
                st.error(f"Authentication failed: {e}")
        else:
            st.error("Please enter both username and password.")

# === Main App Logic (after successful login) ===
else:
    # --- Define Preset Weight Profiles ---
    PRESET_OPTIONS = {
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
            'Distance': 0.25,
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
            'Distance': 0.07,
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
            'Distance': 0.07,
            'Weighted Penetration Score': 0.05,
            'Customer Profile Match Score': 0.10
        }
    }

    # --- Streamlit page settings ---
    st.set_page_config(page_title="Mail Shark Geocode Scoring Tool", layout="wide")
    st.title("📬 Mail Shark Geocode Scoring Tool")

    # --- Select scoring mode from sidebar ---
    preset_choice = st.sidebar.selectbox("Select Scoring Mode", list(PRESET_OPTIONS.keys()))
    DEFAULT_WEIGHTS = PRESET_OPTIONS.get(preset_choice, {})

    # Fallback weights if "Manual" selected or preset is empty
    if not DEFAULT_WEIGHTS:
        DEFAULT_WEIGHTS = {
            '$ Income': 0.20,
            '$ Home Value': 0.15,
            'Owner Occupied': 0.15,
            'Median Year Structure Built': 0.05,
            'House Count': 0.10,
            'House Penetration%': 0.10,
            '$ Total Spend': 0.10,
            'Total Visits': 0.05,
            '$ Average Order': 0.05,
            'Distance': 0.05,
            'Weighted Penetration Score': 0.10,
            'Customer Profile Match Score': 0.10
        }

    # --- Upload penetration report file ---
    uploaded_file = st.file_uploader("Upload your Penetration Report (CSV or XLSX):", type=['csv', 'xlsx'])

    if uploaded_file:


        # Load the file into DataFrame
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.lower().endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        # File validation
        if preset_choice == "Home SRVCS Acquisition (No History)":
            required_columns = ['$ Income', '$ Home Value', 'Owner Occupied', 'Median Year Structure Built', 'Distance']
            file_columns = df.columns
            if not all(col in file_columns for col in required_columns):
                st.error(f"File is missing required columns for preset '{preset_choice}': {required_columns}")
                st.stop()
        else:
            required_columns = ['$ Income', '$ Home Value', 'Owner Occupied', 'Median Year Structure Built',
                                'House Count', 'House Penetration%', '$ Total Spend', 'Total Visits',
                                '$ Average Order', 'Distance', 'Selected']
            file_columns = df.columns
            if not all(col in file_columns for col in required_columns):
                st.error(f"File is missing required columns: {required_columns}")
                st.stop()
        st.write("✅ File loaded:", uploaded_file.name)
        st.write("Cleaning the file and forcing numbers...")

        # Convert currency fields from strings to floats
        currency_columns = ['$ Income', '$ Home Value', '$ Total Spend', '$ Average Order']
        for col in currency_columns:
            if col in df.columns:
                df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

        # --- Weight sliders and filters ---
        st.sidebar.header("Adjust Weights & Filters")
        weights = {}
        for key in DEFAULT_WEIGHTS.keys():
            weights[key] = st.sidebar.slider(f"{key} Weight", 0.0, 1.0, DEFAULT_WEIGHTS[key], 0.01)

        st.sidebar.header("Fail Filter Thresholds")
        min_income = st.sidebar.number_input("Min Household Income ($)", value=40000)
        max_distance = st.sidebar.number_input("Max Distance (miles)", value=50)
        min_owner = st.sidebar.number_input("Min Owner Occupied (%)", value=70)
        max_penetration = st.sidebar.number_input("Max House Penetration (%)", value=100.0)

        # --- Select profile match mode ---
        st.sidebar.header("Customer Profile Mode")
        if preset_choice == "Home SRVCS Acquisition (No History)":
            profile_mode = "Fixed Standard"
            st.sidebar.info("Profile Mode set to 'Fixed Standard' for this preset.")
        else:
            profile_mode = st.sidebar.radio("Customer Profile Mode", label_visibility="collapsed", options=["Dynamic (from file)", "Fixed Standard"])
            if profile_mode == "Fixed Standard":
                st.sidebar.header("Fixed Standard Values")
                ideal_income = st.sidebar.number_input("Ideal Income", value=65000)
                ideal_home_value = st.sidebar.number_input("Ideal Home Value", value=180000)
                ideal_owner = st.sidebar.number_input("Ideal Owner Occupied %", value=75)
                ideal_year_built = st.sidebar.number_input("Ideal Median Year Built", value=1995)
                ideal_distance = st.sidebar.number_input("Ideal Distance (miles for Profile Match)", value=25)
            else:
                # Dynamic profile based on existing customers in file
                st.sidebar.header("Dynamic Profile (from file)")
                st.sidebar.info("Ideal values computed from top 25% House Count in the file.")
                cutoff = df['House Count'].quantile(0.75)
                top = df[df['House Count'] >= cutoff]
                ideal_income = top['$ Income'].mean()
                ideal_home_value = top['$ Home Value'].mean()
                ideal_owner = top['Owner Occupied'].mean()
                ideal_year_built = top['Median Year Structure Built'].mean()
                ideal_distance = top['Distance'].mean()

        # --- Score generation trigger ---
        if st.button("🚀 Generate Scores & Report"):
            working = df.copy()
            winsor_audits = []  # collect per-column audit dicts
            id_series = working['Geocode'] if 'Geocode' in working.columns else None
            # Normalize selected fields with adaptive winsorization
            for col in ['$ Income', '$ Home Value', 'Owner Occupied', 'Median Year Structure Built',
                        'House Count', 'House Penetration%', '$ Total Spend', 'Total Visits',
                        '$ Average Order', 'Distance']:
                if col not in working.columns:
                    continue

                norm, audit = adaptive_minmax_iqr(working[col], col_name=col, id_series=id_series)
                winsor_audits.append(audit)

                if col == 'Distance':
                    norm = 1 - norm  # Invert distance (closer is better)

                working[f"{col}_Norm"] = norm


            # Calculate Weighted Penetration Score with adaptive winsorization
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
            pairs = [
                ('$ Income', ideal_income),
                ('$ Home Value', ideal_home_value),
                ('Median Year Structure Built', ideal_year_built),
                ('Distance', ideal_distance)
            ]

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
            for key in DEFAULT_WEIGHTS.keys():
                if weights[key] > 0:
                    if key == 'Customer Profile Match Score' and key in working.columns:
                        score += working[key] * weights[key]
                    elif key == 'Weighted Penetration Score' and 'Weighted Penetration Score_Norm' in working.columns:
                        score += working['Weighted Penetration Score_Norm'] * weights[key]
                    elif f"{key}_Norm" in working.columns:
                        score += working[f"{key}_Norm"] * weights[key]
            working['Composite Score'] = score

            # Flagging high penetration & filtering
            working['Penetration Flag'] = np.where(
                working['House Penetration%'] > max_penetration,
                "⚠️ Above Max Penetration", ""
            )
            working['Status'] = np.where(
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

            # --- Export results as Excel ---

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
                if a["has_outliers"] and a["clipped_row_indices"]:
                    for rid, orig, new in zip(
                        a["clipped_row_ids"], 
                        a["clipped_original_values"], 
                        a["clipped_new_values"]
                    ):
                        detail_rows.append({
                            "Column": a["column"],
                            "Geocode": rid,
                            "Row Index": a["clipped_row_indices"][a["clipped_row_ids"].index(rid)] if a["clipped_row_ids"] else None,
                            "Original Value": orig,
                            "Clipped Value": new
                        })
            winsor_details_df = pd.DataFrame(detail_rows)
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                # Ranked data
                working.to_excel(writer, index=False, sheet_name='Ranked Geocodes')

                # Base summary (your existing block)
                base_summary = pd.DataFrame({
                    "Note": ["Profile Mode Used:"],
                    "Value": [profile_mode],
                    "Ideal Income": [ideal_income],
                    "Ideal Home Value": [ideal_home_value],
                    "Ideal Owner Occupied": [ideal_owner],
                    "Ideal Median Year Built": [ideal_year_built],
                    "Ideal Distance": [ideal_distance]
                })

                # Write base summary at the top
                base_summary.to_excel(writer, sheet_name='Summary', index=False, startrow=0)

                # Write winsor summary below it, with a blank row in between
                startrow = len(base_summary) + 2
                if not audit_summary_df.empty:
                    audit_summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=startrow)

                # Optional detailed sheet
                if not winsor_details_df.empty:
                    winsor_details_df.to_excel(writer, sheet_name='Winsorized Rows', index=False)

            st.download_button(
                label="📥 Download Ranked Excel",
                data=towrite,
                file_name=f"{uploaded_file.name.split('.')[0]}_Ranked.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
