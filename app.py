import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def safe_search(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

def extract_info_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    data = {}
    data['Operator'] = safe_search(r"Operator\s+(.*)", text)
    data['Rig Name'] = safe_search(r"Rig Name\s+(.*)", text)
    data['Well Name'] = safe_search(r"Well Name\s+(.*)", text)
    data['Date'] = safe_search(r"Date\s+(\d{4}-\d{2}-\d{2})", text)
    data['MD (ft)'] = safe_search(r"Depth \(MD/TVD\)\s+([\d.]+)", text)
    data['Bit Size'] = safe_search(r"Bit Size\s+([\d.]+)", text)
    data['Drilling Hrs'] = safe_search(r"Drilling\s+(\d+)", text)
    data['Total Circ'] = safe_search(r"Total Circ\s+([\d.]+)", text)
    data['LGS%'] = safe_search(r"LGS\s*/\s*HGS\s*%\s*([\d.]+)", text)
    data['Base Oil'] = safe_search(r"Base\s+([\d.]+)", text)
    data['Water'] = safe_search(r"Drill Water\s+([\d.]+)", text)
    data['Barite'] = safe_search(r"Barite\s+([\d.]+)", text)
    data['Chemical'] = safe_search(r"Chemicals\s+([\d.]+)", text)
    data['Reserve'] = safe_search(r"Reserve\s+\*\s+([\d.]+)", text)

    loss_match = re.search(r"SCE\s+([\d.]+).*?Other\s+([\d.]+)", text, re.DOTALL)
    if loss_match:
        data['Losses'] = float(loss_match.group(1)) + float(loss_match.group(2))
    else:
        data['Losses'] = 0

    data['Mud Flow'] = safe_search(r"gpm\s+([\d.]+)", text)
    data['PV'] = safe_search(r"PV\s+@.*?([\d.]+)", text)
    data['YP'] = safe_search(r"YP\s+lb/100ft²\s+([\d.]+)", text)
    data['Mud Weight'] = safe_search(r"Density\s+@.*?([\d.]+\s*@\s*[\d.]+)", text)

    return data

def to_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def simulate_label(row):
    return int(
        to_float(row['LGS%']) > 10 or
        to_float(row['Losses']) > 200 or
        to_float(row['PV']) > 35
    )

st.title("📄 Drilling Fluid Report Extractor + KPI Dashboard + ML Degradation Predictor")

uploaded_files = st.file_uploader("Upload Daily Drilling Fluid PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    records = []
    for file in uploaded_files:
        try:
            record = extract_info_from_pdf(file)
            records.append(record)
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")

    if records:
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        st.success("✅ Data Extracted!")
        st.dataframe(df)

        # --- Filter Controls ---
        with st.sidebar:
            st.header("🔍 Filters")
            well_options = df['Well Name'].dropna().unique().tolist()
            selected_wells = st.multiselect("Select Well(s)", well_options, default=well_options)
            date_range = st.date_input("Select Date Range", [df['Date'].min(), df['Date'].max()])

        df = df[df['Well Name'].isin(selected_wells)]
        df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]


        df['Degraded'] = df.apply(simulate_label, axis=1)

        # Convert necessary fields to float
        for col in ['LGS%', 'PV', 'YP', 'Mud Flow', 'Losses', 'Base Oil', 'Water', 'Chemical', 'Total Circ', 'Drilling Hrs', 'MD (ft)']:
            df[col] = df[col].apply(to_float)

        # Calculate derived KPIs
        df['Total SCE'] = df['Losses']
        df['Discard Ratio'] = df['Total SCE'] / df['Total Circ']
        df['Total Dilution'] = df[['Base Oil', 'Water', 'Chemical']].sum(axis=1)
        df['Dilution Ratio'] = df['Total Dilution'] / df['Total Circ']
        df['DSRE%'] = (df['Total Dilution'] / (df['Total Dilution'] + df['Total SCE'])) * 100
        df['ROP'] = df['MD (ft)'] / df['Drilling Hrs']
        df['Ave Temp'] = df['Mud Weight'].str.extract(r'@\s*(\d+)').astype(float)
        df['Mud Cutting Ratio'] = df['LGS%'] / df['Total Circ'] * 100
        df['Solid Generate'] = df['LGS%'] * df['Total Circ'] / 100

        # ML Model
        features = ['LGS%', 'PV', 'YP', 'Mud Flow', 'Losses']
        X = df[features]
        y = df['Degraded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        df['Predicted Degradation'] = clf.predict(X)

        st.subheader("🧠 ML-Based Degradation Prediction")
        st.dataframe(df[['Date', 'LGS%', 'PV', 'YP', 'Losses', 'Predicted Degradation']])

        st.text("📊 Model Performance on Holdout Data")
        y_pred = clf.predict(X_test)
        st.text(classification_report(y_test, y_pred))

        # Charts in tabs
        st.subheader("📊 KPI Dashboard")
        tab1, tab2, tab3 = st.tabs(["Performance KPIs", "Dilution & Losses", "Solids"])

        
    with tab1:    st.altair_chart(alt.Chart(df).mark_line(point=True).encode(
x='Date:T',
        y='ROP:Q',
        tooltip=['Date', 'ROP']
    ).properties(title="ROP Over Time", height=300), use_container_width=True)

    st.altair_chart(alt.Chart(df).mark_area().encode(
        x='Date:T',
        y='Mud Flow:Q',
        tooltip=['Date', 'Mud Flow']
    ).properties(title="Mud Flow Trend", height=300), use_container_width=True)

    st.altair_chart(alt.Chart(df).mark_line().encode(
        x='Date:T',
        y='Ave Temp:Q',
        tooltip=['Date', 'Ave Temp']
    ).properties(title="Average Temperature", height=300), use_container_width=True)


        
    with tab2:    st.altair_chart(alt.Chart(df).mark_bar().encode(
x='Date:T',
        y='Total Dilution:Q',
        tooltip=['Date', 'Total Dilution']
    ).properties(title="Total Dilution", height=300), use_container_width=True)

    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x='Date:T',
        y='Total SCE:Q',
        tooltip=['Date', 'Total SCE']
    ).properties(title="Total SCE", height=300), use_container_width=True)

    st.altair_chart(alt.Chart(df).mark_line().encode(
        x='Date:T',
        y='DSRE%:Q',
        tooltip=['Date', 'DSRE%']
    ).properties(title="DSRE%", height=300), use_container_width=True)


    with tab3:            st.area_chart(df.set_index('Date')[['Solid Generate', 'Mud Cutting Ratio']])
        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "fluid_kpis_ml.csv", "text/csv")
