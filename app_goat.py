import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from data_cleaning import clean_data, remove_duplicates, remove_outliers
from sklearn.metrics import confusion_matrix, classification_report

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme Detection ---
theme = st.get_option("theme.base")
if theme == "dark":
    background_color = "#0e1117"
    font_color = "#FFFFFF"
else:
    background_color = "#FFFFFF"
    font_color = "#000000"

# --- Main Header Layout ---
st.markdown(
    f"""
    <div style="background-color:{background_color}; padding: 20px; border-radius: 10px;">
        <h1 style="color:{font_color}; text-align: center;">🚀 Predictive Maintenance Dashboard</h1>
        <p style="color:{font_color}; text-align: center; font-size:18px;">
            Enhance reliability. Predict failures. Optimize performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Divider Line ---
st.markdown("---")


# --- Advisory Priority Constants ---
ADVISORY_PRIORITY = {
    "very high fired": {"color": "red", "priority": 1, "position": "high"},
    "long high fired": {"color": "yellow", "priority": 2, "position": "high"},
    "high fired": {"color": "orange", "priority": 3, "position": "high"},
    "very low fired": {"color": "red", "priority": 1, "position": "low"},
    "long low fired": {"color": "yellow", "priority": 2, "position": "low"},
    "low fired": {"color": "orange", "priority": 3, "position": "low"}
}

# --- Load model utilities ---
@st.cache_resource(show_spinner=False)
def get_available_models():
    model_files = [
        f
        for f in os.listdir("saved_models")
        if f.endswith(".pkl") and not any(x in f for x in ["scaler", "label_encoder", "features"])
    ]
    model_map = {}
    for f in model_files:
        try:
            sheet, model = f.replace(".pkl", "").rsplit("_", 1)
            model_map.setdefault(sheet, []).append(model)
        except ValueError:
            continue
    return model_map

@st.cache_resource(show_spinner=False)
def load_model(sheet, model_name):
    model_path = f"saved_models/{sheet}_{model_name}.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

@st.cache_resource(show_spinner=False)
def load_label_encoder(sheet):
    le_path = f"saved_models/label_encoders/{sheet}_label_encoder.pkl"
    if os.path.exists(le_path):
        return joblib.load(le_path)
    return None

@st.cache_resource(show_spinner=False)
def load_scaler(sheet):
    scaler_path = f"saved_models/scalers/{sheet}_scaler.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

# --- Match features dynamically ---

def match_features(expected_keywords, columns):
    """Match columns based on keywords regardless of serial numbers"""
    matched = []
    for keyword in expected_keywords:
        # Find columns containing the keyword (case insensitive)
        candidates = [col for col in columns if keyword.lower() in col.lower()]
        
        if not candidates:
            st.warning(f"⚠️ No column found matching keyword: '{keyword}'")
            continue
            
        # Select the first matching column
        selected = candidates[0]
        matched.append(selected)
        
        # Log the matching for transparency
        st.info(f"Matched '{keyword}' with column: '{selected}'")
    
    return matched

# --- Classify Advisory Type with Priority ---
def classify_advisory(text):
    if pd.isnull(text) or str(text).strip().lower() == "normal":
        return None, None, None
    
    text_lower = str(text).lower()
    
    # Find all matching advisory types
    matches = []
    for advisory_type, props in ADVISORY_PRIORITY.items():
        if advisory_type in text_lower:
            matches.append((props["priority"], advisory_type))
    
    # Return the highest priority match (lowest priority number)
    if matches:
        highest_priority = min(matches, key=lambda x: x[0])
        advisory_type = highest_priority[1]
        return ADVISORY_PRIORITY[advisory_type]["color"], ADVISORY_PRIORITY[advisory_type]["position"], advisory_type
    
    return None, None, None

# --- UI: Sidebar model selection ---
available_models = get_available_models()
if not available_models:
    st.error("❌ No trained models found in 'saved_models' directory.")
    st.stop()

sheet = st.sidebar.selectbox("📄 Select Data Sheet", sorted(available_models.keys()))
model_name = st.sidebar.selectbox("🧠 Select Model", sorted(available_models[sheet]))

# In your main UI flow, after model selection but before file upload processing
try:
    model = load_model(sheet, model_name)
    scaler = load_scaler(sheet)
    label_encoder = load_label_encoder(sheet)
    
    # Get the original feature names the model was trained with
    try:
        original_features = joblib.load(f"saved_models/features/{sheet}_features.pkl")  # Updated path
    except:
        original_features = None
        st.warning("Original training features not available")
    
except Exception as e:
    st.error(f"❌ Error loading model components: {str(e)}")
    st.stop()

# --- UI: Upload file ---
st.markdown("### 📂 Upload Your Sensor Data")
uploaded_file = st.file_uploader("📤 Upload Excel File with Multiple Sheets", type=["xlsx"])
if uploaded_file:
    st.markdown("### 🔍 Data Cleaning and Preprocessing")
    with st.spinner("Loading and processing data..."):
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        if sheet not in all_sheets:
            st.warning(f"⚠️ Sheet '{sheet}' not found.")
            st.stop()

        df = all_sheets[sheet]
        
        # --- CLEAN DATA ---
        df = clean_data(df)
        diag_col = next((col for col in df.columns if "Diagnostic Advisory Indication" in col), None)
        if diag_col:
            df = remove_duplicates(df, diag_col)
            df = remove_outliers(df, diag_col)

        # --- Match features dynamically ---
        expected_keywords = ["Actual", "Estimate", "Residual"]
        training_features = match_features(expected_keywords, df.columns)

        # Verify we found all required features
        if len(training_features) != len(expected_keywords):
            missing = set(expected_keywords) - {k.lower() for k in training_features}
            st.error(f"❌ Missing required features: {missing}")
            st.stop()

        st.subheader("🔍 Matched Features")
        st.write(training_features)

        # --- Feature Alignment ---
        if original_features:
            st.info("🔧 Aligning features with original training format...")
            
            # Create mapping between current and original features
            feature_map = {}
            for orig_feature in original_features:
                for keyword in expected_keywords:
                    if keyword.lower() in orig_feature.lower():
                        # Find matching column in current data
                        matches = [col for col in training_features if keyword.lower() in col.lower()]
                        if matches:
                            feature_map[matches[0]] = orig_feature
                        break
            
            # Verify alignment
            if len(feature_map) == len(original_features):
                st.success("✅ Features successfully aligned")
            else:
                st.warning("⚠️ Partial feature alignment - some original features not matched")

        # --- Scale and Predict ---
        X_input = df[training_features]
        # Always initialize scaled input
        X_input_scaled = X_input.values  # default unscaled

        if scaler:
            try:
                # Try aligning features if possible
                if original_features and len(feature_map) == len(original_features):
                    X_input_aligned = X_input.rename(columns=feature_map)[original_features]
                    st.info(f"Aligned features: {list(X_input_aligned.columns)}")
                    X_input_scaled = scaler.transform(X_input_aligned)
                else:
                    st.warning("⚠️ Using unmatched features - performance may vary")
                    X_input_scaled = scaler.transform(X_input)
                    
            except Exception as e:
                st.error(f"❌ Error scaling input: {e}")
                st.stop()


        try:
            prediction = model.predict(X_input_scaled)

            # Decode predictions
            if label_encoder:
                try:
                    decoded_predictions = label_encoder.inverse_transform(prediction)
                    df["Predicted"] = decoded_predictions
                    
                    with st.expander("🔤 DECODED PREDICTIONS", expanded=False):
                        st.write("Label Classes:", list(label_encoder.classes_))
                        st.write("First 5 Decoded:", decoded_predictions[:5])
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not decode predictions: {e}. Showing numeric labels.")
                    df["Predicted"] = prediction
            else:
                # Fallback mapping
                LABEL_MAPPING = {
                    0: "No Failure Indication",
                    1: "Early Failure Indication", 
                    2: "Severe Failure Indication"
                }
                df["Predicted"] = [LABEL_MAPPING.get(p, f"Unknown ({p})") for p in prediction]
                
                with st.expander("🗂️ LABEL MAPPING APPLIED", expanded=False):
                    st.write("Mapping Used:", LABEL_MAPPING)
                    st.write("Sample Mapped:", df["Predicted"].head().tolist())
                    
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

        # ===== POST-PROCESSING DEBUG =====
        with st.expander("🛠️ SPECIAL CASE HANDLING", expanded=False):
            # Handle special cases
            df["Predicted"] = df["Predicted"].astype(str)
            advisory_col = next(
                (col for col in df.columns if "Advisory Indication" in col and "Diagnostic" not in col), None
            )
            
            st.write("**Advisory Column Found:**", advisory_col)
            if advisory_col:
                st.write("Advisory Value Counts:", df[advisory_col].value_counts())
                
                mask_pred_nan = df["Predicted"].str.lower().isin(["nan", "none", "null", ""])
                mask_advisory_normal = df[advisory_col].str.strip().str.lower() == "normal"
                
                st.write("**Before Handling:**")
                st.write("NaN Predictions:", mask_pred_nan.sum())
                st.write("Normal Advisories:", mask_advisory_normal.sum())
                st.write("Matches for No Failure:", (mask_pred_nan & mask_advisory_normal).sum())
                
                df.loc[mask_pred_nan & mask_advisory_normal, "Predicted"] = "No Failure Indication"
                
                st.write("**After Handling:**")
                st.write(df["Predicted"].value_counts(dropna=False))
            else:
                st.warning("No advisory column found for special case handling")

        # After predictions are complete, before any validation or plotting
        if diag_col:
            df[diag_col] = df[diag_col].fillna("No Failure Indication")
            df["Predicted"] = df["Predicted"].fillna("No Failure Indication")  # <-- THIS IS WHAT WAS MISSING

            # Also fix if "Predicted" has literal strings like "nan", "None", "null", ""
            df["Predicted"] = df["Predicted"].replace(["nan", "None", "null", ""], "No Failure Indication")
            df[diag_col] = df[diag_col].replace(["nan", "None", "null", ""], "No Failure Indication")


        # Then inside the expander:
        with st.expander("✅ FINAL PREDICTION VALIDATION", expanded=True):
            st.write("**Final Prediction Distribution:**")
            st.write(df["Predicted"].value_counts(normalize=True))
            if diag_col:
                st.write("**Confusion Matrix Preview:**")
                st.dataframe(pd.crosstab(df[diag_col], df["Predicted"], margins=True, margins_name="Total"))

        # --- Summary Panel ---
        st.markdown("### 📊 Quick Summary")

        total_data = len(df)

        # Failure = anything NOT "No Failure Indication"
        total_failures = df["Predicted"].str.strip().str.lower().ne("no failure indication").sum()

        accuracy = None

        # If ground truth available
        if diag_col:
            y_true = df[diag_col].astype(str).fillna("No Failure Indication")
            y_pred = df["Predicted"].astype(str).fillna("No Failure Indication")
            correct = (y_true == y_pred).sum()
            accuracy = correct / total_data * 100

        cols = st.columns(3)
        with cols[0]:
            st.metric("📄 Total Data Points", total_data)
        with cols[1]:
            st.metric("⚡ Failures Detected", total_failures)
        with cols[2]:
            if accuracy is not None:
                st.metric("🎯 Approx. Accuracy", f"{accuracy:.2f}%")
            else:
                st.metric("🎯 Approx. Accuracy", "N/A")


        st.markdown("### 🔮 Predictive Analysis Results")
        # --- TIMESTAMP ---
        timestamp_col = df.columns[0]
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col)

        sensor_col = training_features[0]
        estimate_col = next((col for col in training_features if "Estimate" in col), None)
        residual_col = next((col for col in training_features if "Residual" in col), None)

        # --- Plotly Graph: Actual vs Estimate ---
        st.markdown("#### 📈 Actual vs Estimate with Advisories")
        fig = go.Figure()
        
        # 1. Plot Actual and Estimate lines
        fig.add_trace(
            go.Scatter(x=df[timestamp_col], y=df[sensor_col], mode="lines", name="Actual", line=dict(width=2))
        )
        if estimate_col:
            fig.add_trace(
                go.Scatter(x=df[timestamp_col], y=df[estimate_col], mode="lines", name="Estimate", line=dict(dash="dash"))
            )

        # Calculate positions
        y_max = df[sensor_col].max()
        y_min = df[sensor_col].min()
        offset = 0.05 * (y_max - y_min)

        # 2. Add Advisory Markers (circle-open) with priority handling
        if advisory_col:
            advisory_data = df[[timestamp_col, advisory_col]].dropna()
            
            # Classify each advisory with priority
            advisory_data["color"], advisory_data["position"], advisory_data["type"] = zip(
                *advisory_data[advisory_col].apply(classify_advisory)
            )
            advisory_data = advisory_data.dropna()
            
            # For each timestamp, keep only the highest priority advisory
            advisory_data = advisory_data.sort_values(timestamp_col)
            advisory_data = advisory_data.groupby(timestamp_col).apply(
                lambda x: x.sort_values("type", key=lambda y: y.map(
                    {k: v["priority"] for k, v in ADVISORY_PRIORITY.items()}
                )).head(1)
            ).reset_index(drop=True)
            
            # Position markers
            advisory_data["y_coord"] = advisory_data["position"].map(
                lambda p: y_max + offset if p == "high" else y_min - offset
            )
            
            # Add traces for each advisory type
            for advisory_type, props in ADVISORY_PRIORITY.items():
                subset = advisory_data[advisory_data["type"] == advisory_type]
                if not subset.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=subset[timestamp_col],
                            y=subset["y_coord"],
                            mode="markers",
                            name=f"Advisory ({advisory_type.title()})",
                            marker=dict(
                                color=props["color"],
                                size=10,
                                symbol="circle-open",
                                line=dict(width=1.5)
                            ),
                            hovertext=subset[advisory_col],
                        )
                    )

        # 3. Add Diagnostic Markers (x markers) based on Predicted (filtered)
        if "Predicted" in df.columns:
            diag_data = df[[timestamp_col, "Predicted"]].dropna()
            # Only plot diagnostics that are NOT 'No Failure Indication'
            diag_data = diag_data[~diag_data["Predicted"].str.lower().str.contains("no failure indication")]

            if not diag_data.empty:
                diag_data["marker_color"] = diag_data["Predicted"].apply(
                    lambda x: "grey" if "early failure" in x.lower() else "black"
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=diag_data[timestamp_col],
                        y=[y_max + 2 * offset] * len(diag_data),
                        mode="markers",
                        name="Diagnostic",
                        marker=dict(
                            size=12,
                            symbol="x",
                            color=diag_data["marker_color"],
                            line=dict(width=1.5)
                        ),
                        hovertext=diag_data["Predicted"],
                    )
                )

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Sensor Reading",
            height=600,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Residual Plot with Advisory Thresholds ---
        if residual_col:
            st.markdown("#### 📉 Residual Plot with Advisory Thresholds")
            residual_fig = go.Figure()
            
            # 1. Plot Residual line
            residual_fig.add_trace(
                go.Scatter(
                    x=df[timestamp_col],
                    y=df[residual_col],
                    mode="lines",
                    name="Residual",
                    line=dict(color="red", width=2)
                )
            )
            
            y_resid_max = df[residual_col].max()
            y_resid_min = df[residual_col].min()
            offset_resid = 0.05 * (y_resid_max - y_resid_min)
            
            # 2. Add Advisory Markers (x markers) with priority
            if advisory_col:
                advisory_points = df.dropna(subset=[advisory_col, residual_col]).copy()
                
                # Assign marker properties based on advisory type
                def get_marker_props(text):
                    text = str(text).lower()
                    if "very high fired" in text:
                        return {"color": "darkred", "position": "top", "priority": 1}
                    elif "high fired" in text or "long high fired" in text:
                        return {"color": "red", "position": "top", "priority": 2}
                    elif "very low fired" in text:
                        return {"color": "darkred", "position": "bottom", "priority": 1}
                    elif "low fired" in text or "long low fired" in text:
                        return {"color": "red", "position": "bottom", "priority": 2}
                    return None
                
                advisory_points["marker_props"] = advisory_points[advisory_col].apply(get_marker_props)
                advisory_points = advisory_points.dropna(subset=["marker_props"])
                
                # For each timestamp, keep only the highest priority advisory
                advisory_points = advisory_points.sort_values(timestamp_col)
                advisory_points = advisory_points.groupby(timestamp_col).apply(
                    lambda x: x.sort_values("marker_props", key=lambda y: y.apply(lambda z: z["priority"])).head(1)
                ).reset_index(drop=True)
                
                # Position markers
                advisory_points["marker_y"] = advisory_points["marker_props"].apply(
                    lambda x: y_resid_max + offset_resid if x["position"] == "top" else y_resid_min - offset_resid
                )
                
                # Add markers for each advisory type
                marker_types = [
                    ("Very High Fired", "darkred", "top"),
                    ("High Fired", "red", "top"),
                    ("Very Low Fired", "darkred", "bottom"),
                    ("Low Fired", "red", "bottom")
                ]
                
                for adv_type, color, pos in marker_types:
                    subset = advisory_points[
                        (advisory_points["marker_props"].apply(lambda x: x["color"] == color)) & 
                        (advisory_points["marker_props"].apply(lambda x: x["position"] == pos))
                    ]
                    if not subset.empty:
                        residual_fig.add_trace(
                            go.Scatter(
                                x=subset[timestamp_col],
                                y=subset["marker_y"],
                                mode="markers",
                                name=f"{adv_type} Advisory",
                                marker=dict(
                                    symbol="x",
                                    size=12,
                                    color=color,
                                    line=dict(width=0)  # No border for cleaner look
                                ),
                                hovertext=subset[advisory_col],
                            )
                        )
                
                # 3. Add Threshold Lines
                thresholds = {
                    "Very High Fired": {
                        "color": "darkred",
                        "value": advisory_points[advisory_points[advisory_col].str.contains("Very High Fired", case=False)][residual_col].min()
                    },
                    "High Fired": {
                        "color": "red",
                        "value": advisory_points[advisory_points[advisory_col].str.contains("High Fired", case=False)][residual_col].min()
                    },
                    "Very Low Fired": {
                        "color": "darkred",
                        "value": advisory_points[
                            (advisory_points[advisory_col].str.contains("Very Low Fired", case=False)) &
                            (advisory_points[residual_col] < 0)
                        ][residual_col].max()  # Gets largest negative value (closest to zero)
                    },
                    "Low Fired": {
                        "color": "red",
                        "value": advisory_points[
                            (advisory_points[advisory_col].str.contains("Low Fired", case=False)) &
                            (advisory_points[residual_col] < 0)
                        ][residual_col].max()  # Gets largest negative value (closest to zero)
                    }
                }
                
                for label, props in thresholds.items():
                    if pd.notnull(props["value"]):
                        residual_fig.add_trace(
                            go.Scatter(
                                x=df[timestamp_col],
                                y=[props["value"]] * len(df),
                                name=f"{label} Threshold",
                                line=dict(
                                    color=props["color"],
                                    dash="dash",
                                    width=2
                                ),
                                hoverinfo="skip"
                            )
                        )
            
            residual_fig.update_layout(
                xaxis_title="Timestamp",
                yaxis_title="Residual",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(residual_fig, use_container_width=True)

        # --- Advisory Indication Distribution (Keyword-Based) ---
        if advisory_col:
            st.markdown("#### 📊 Advisory Indication Distribution")

            # 1️⃣ Calculate keyword-based advisory counts
            advisory_keywords = list(ADVISORY_PRIORITY.keys())
            keyword_counts = {k: 0 for k in advisory_keywords}

            for value in df[advisory_col].dropna():
                text = str(value).lower()
                for keyword in advisory_keywords:
                    if keyword in text:
                        keyword_counts[keyword] += 1

            keyword_counts_df = pd.DataFrame({
                "Advisory Type": list(keyword_counts.keys()),
                "Count": list(keyword_counts.values())
            })
            keyword_counts_df["Percentage"] = (keyword_counts_df["Count"] / keyword_counts_df["Count"].sum() * 100).round(2)
            keyword_counts_df = keyword_counts_df.sort_values(by="Count", ascending=False)

            # 2️⃣ Display table (based on keyword counts)
            st.dataframe(
                keyword_counts_df.style.format({'Percentage': '{:.2f}%'}),
                height=min(300, 35 + 35 * len(keyword_counts_df))  # Dynamic height
            )

            # 3️⃣ Visualization options
            with st.expander("📈 Visualization Options", expanded=True):
                tab1, tab2 = st.tabs(["Bar Chart", "Trend Over Time"])

                # --- Tab 1: Bar Chart ---
                with tab1:
                    fig = px.bar(
                        keyword_counts_df,
                        y="Advisory Type",
                        x="Count",
                        orientation='h',
                        title="Advisory Type Distribution (Keyword-Based)",
                        text="Count",
                        color="Advisory Type",
                        color_discrete_map={k: v['color'] for k, v in ADVISORY_PRIORITY.items()}
                    )
                    fig.update_layout(
                        showlegend=False,
                        yaxis_title="Advisory Type",
                        xaxis_title="Count",
                        height=400 + 20 * len(keyword_counts_df)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # --- Tab 2: Trend Over Time ---
                with tab2:
                    if len(df[timestamp_col].dropna()) > 1:
                        # Create new DataFrame for trend
                        advisory_trend = pd.DataFrame({timestamp_col: df[timestamp_col]})

                        for keyword in advisory_keywords:
                            advisory_trend[keyword] = df[advisory_col].str.contains(keyword, case=False, na=False).astype(int)

                        advisory_daily = advisory_trend.groupby(
                            pd.Grouper(key=timestamp_col, freq='D')
                        ).sum()

                        fig = px.line(
                            advisory_daily,
                            title="Daily Advisory Trends (Keyword-Based)",
                            labels={'value': 'Count', 'variable': 'Advisory Type'}
                        )
                        fig.update_layout(
                            height=400,
                            xaxis_title="Date",
                            yaxis_title="Count",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough date data to show time trends.")

            # 4️⃣ Key statistics based on keywords
            st.markdown("##### 📌 Key Statistics")
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Advisories (Sum)", int(keyword_counts_df["Count"].sum()))
            with cols[1]:
                if not keyword_counts_df.empty:
                    most_frequent = keyword_counts_df.iloc[0]
                    st.metric(
                        "Most Frequent Advisory",
                        most_frequent["Advisory Type"],
                        delta=f"{most_frequent['Percentage']:.2f}%"
                    )
                else:
                    st.metric("Most Frequent Advisory", "None")
            with cols[2]:
                if len(df[timestamp_col].dropna()) > 0:
                    date_range = f"{df[timestamp_col].min().strftime('%Y-%m-%d')} to {df[timestamp_col].max().strftime('%Y-%m-%d')}"
                    st.metric("Time Range", date_range)



        # --- Prediction Results Preview ---
        st.markdown("#### 📄 Prediction Result Preview")
        
        preview_cols = [timestamp_col] + training_features
        # Add advisory_col if it exists
        if advisory_col:
            preview_cols.append(advisory_col)
        # Add "Predicted" column at the very end
        preview_cols.append("Predicted")
        # Select the columns in the desired order
        df_preview = df[preview_cols]
        # Sample 25 random rows for preview (if df has less than 25 rows, it will just show all)
        df_sample = df_preview.sample(n=25, random_state=42) 
        # Display the dataframe
        st.dataframe(df_sample)


        # --- Diagnostic Prediction Distribution ---
        st.markdown("#### 📊 Diagnostic Prediction Distribution")
        pred_counts = df["Predicted"].value_counts().reset_index()
        pred_counts.columns = ["Diagnostic Class", "Count"]
        pred_counts["Percentage"] = (pred_counts["Count"] / pred_counts["Count"].sum() * 100).round(2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(pred_counts)
        
        with col2:
            bar_fig = px.bar(
                pred_counts,
                x="Diagnostic Class",
                y="Count",
                text=pred_counts["Percentage"].map(lambda x: f"{x:.2f}%"),
                title="Distribution of Predicted Diagnostics"
            )
            bar_fig.update_traces(textposition="outside")
            bar_fig.update_layout(xaxis_title="Diagnostic Class", yaxis_title="Count")
            st.plotly_chart(bar_fig, use_container_width=True)

        # --- Confusion Matrix and Report ---
        if hasattr(model, "classes_") and diag_col and df[diag_col].nunique() > 1:
            y_true = df[diag_col].astype(str).fillna("No Failure Indication")
            y_pred = df["Predicted"].astype(str).fillna("No Failure Indication")
            
            # Clean labels
            y_true = y_true.replace(["nan", "None", "null", ""], "No Failure Indication")
            y_pred = y_pred.replace(["nan", "None", "null", ""], "No Failure Indication")
            
            unique_labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
            if "No Failure Indication" not in unique_labels:
                unique_labels.append("No Failure Indication")
            unique_labels = sorted(unique_labels)
            
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
            cm_df.index.name = 'Actual'
            cm_df.columns.name = 'Predicted'
            st.subheader("📉 Confusion Matrix")
            st.dataframe(cm_df)

            st.subheader("📋 Classification Report")
            report = classification_report(y_true, y_pred, target_names=unique_labels, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
            
        # --- Download Button ---
        st.markdown("#### ⬇️ Download Predictions")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")

        # --- Manual Real-Time Prediction ---
        st.markdown("### ✍️ Manual Real-Time Prediction")

        with st.expander("🔎 Enter Actual and Estimate Values", expanded=True):
            st.markdown("Fill the sensor readings below to predict diagnostic advisory.")

            with st.form("manual_input_form"):
                # --- Dynamically detect the feature names ---
                actual_col = next(col for col in training_features if "Actual" in col)
                estimate_col = next(col for col in training_features if "Estimate" in col)
                residual_col = next(col for col in training_features if "Residual" in col)

                cols = st.columns(2)
                with cols[0]:
                    actual_value = st.number_input(
                        f"🔵 {actual_col}",
                        value=0.0,
                        step=0.00001,
                        format="%.5f"
                    )
                with cols[1]:
                    estimate_value = st.number_input(
                        f"🟣 {estimate_col}",
                        value=0.0,
                        step=0.00001,
                        format="%.5f"
                    )

                # --- Live residual calculation preview ---
                calculated_residual = actual_value - estimate_value
                st.info(f"🧮 **Calculated Residual ({residual_col})**: `{calculated_residual:.5f}`")

                # --- Submit button ---
                submitted = st.form_submit_button("🔮 Predict Diagnostic")

                if submitted:
                    try:
                        # Build manual input
                        input_data = []
                        for feature in training_features:
                            if feature == actual_col:
                                input_data.append(actual_value)
                            elif feature == estimate_col:
                                input_data.append(estimate_value)
                            elif feature == residual_col:
                                input_data.append(calculated_residual)

                        manual_df = pd.DataFrame([input_data], columns=training_features)

                        # Align manual input if needed
                        if original_features and 'feature_map' in locals():
                            manual_df = manual_df.rename(columns=feature_map)[original_features]

                        manual_scaled = scaler.transform(manual_df) if scaler else manual_df
                        prediction_result = model.predict(manual_scaled)

                        if label_encoder:
                            try:
                                decoded = label_encoder.inverse_transform(prediction_result)[0]
                            except Exception as e:
                                st.warning(f"⚠️ Could not decode prediction: {e}")
                                decoded = prediction_result[0]
                        else:
                            LABEL_MAPPING = {
                                0: "No Failure Indication",
                                1: "Early Failure Indication",
                                2: "Severe Failure Indication"
                            }
                            decoded = LABEL_MAPPING.get(prediction_result[0], f"Unknown ({prediction_result[0]})")

                        # --- Show prediction result nicely ---
                        st.success(f"🎯 **Predicted Diagnostic:** `{decoded}`", icon="✅")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")

