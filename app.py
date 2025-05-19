import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import plotly.express as px
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
    data['Losses'] = float(safe_search(r"SCE\s+([\d.]+)", text) or 0)
    data['Mud Flow'] = safe_search(r"gpm\s+([\d.]+)", text)
    data['Mud Weight'] = safe_search(r"Density\s+@.*?([\d.]+)", text)
    data['Ave Temp'] = safe_search(r"Fl Temp\s+([\d.]+)", text)
    data['PV'] = safe_search(r"PV\s+@.*?([\d.]+)", text)
    data['YP'] = safe_search(r"YP\s+lb/100ft²\s+([\d.]+)", text)
    data['Pump 1 GPM'] = safe_search(r"Pump 1.*?gpm\s+([\d.]+)", text)
    data['Pump 2 GPM'] = safe_search(r"Pump 2.*?gpm\s+([\d.]+)", text)
    data['Pump 3 GPM'] = safe_search(r"Pump 3.*?gpm\s+([\d.]+)", text)
    data['API Screen'] = safe_search(r"API\s+Mesh\s+([\d]+)", text)
    data['Screen Count'] = safe_search(r"Screen Count\s+([\d]+)", text)
    return data

def to_float(val, default=0.0):
    try:
        return float(val)
    except:
        return default

def simulate_label(row):
    return int(
        to_float(row['LGS%']) > 10 or
        to_float(row['Losses']) > 200 or
        to_float(row['PV']) > 35
    )

st.title("🛢️ Drilling Fluid Dashboard + GPM + Recommendations")

uploaded_files = st.file_uploader("Upload Fluid Reports (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    records = []
    for file in uploaded_files:
        try:
            records.append(extract_info_from_pdf(file))
        except Exception as e:
            st.error(f"Failed to parse {file.name}: {e}")

    if records:
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        df['Degraded'] = df.apply(simulate_label, axis=1)

        for col in ['LGS%', 'PV', 'YP', 'Mud Flow', 'Losses', 'Base Oil', 'Water', 'Chemical',
                    'Total Circ', 'Drilling Hrs', 'MD (ft)', 'Mud Weight', 'Ave Temp',
                    'Pump 1 GPM', 'Pump 2 GPM', 'Pump 3 GPM', 'Screen Count']:
            df[col] = df[col].apply(to_float)

        df['Total SCE'] = df['Losses']
        df['Discard Ratio'] = df['Total SCE'] / df['Total Circ']
        df['Total Dilution'] = df[['Base Oil', 'Water', 'Chemical']].sum(axis=1)
        df['Dilution Ratio'] = df['Total Dilution'] / df['Total Circ']
        df['DSRE%'] = (df['Total Dilution'] / (df['Total Dilution'] + df['Total SCE'])) * 100
        df['ROP'] = df['MD (ft)'] / df['Drilling Hrs']
        df['Mud Cutting Ratio'] = df['LGS%'] / df['Total Circ'] * 100
        df['Solid Generate'] = df['LGS%'] * df['Total Circ'] / 100
        df['GPM Total'] = df[['Pump 1 GPM', 'Pump 2 GPM', 'Pump 3 GPM']].sum(axis=1)
        df['GPM/Screen'] = df['GPM Total'] / df['Screen Count'].replace(0, 1)

        st.dataframe(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["ROP & Flow", "Dilution", "Solids", "Recommendations", "Combined"])

        with tab1:
            st.plotly_chart(px.line(df, x='Date', y='ROP', color='Well Name', title='ROP'), use_container_width=True)
            st.plotly_chart(px.line(df, x='Date', y='Mud Flow', color='Well Name', title='Mud Flow'), use_container_width=True)

        with tab2:
            st.plotly_chart(px.bar(df, x='Date', y='Total Dilution', color='Well Name', title='Total Dilution'), use_container_width=True)
            st.plotly_chart(px.line(df, x='Date', y='DSRE%', color='Well Name', title='DSRE%'), use_container_width=True)

        with tab3:
            st.plotly_chart(px.line(df, x='Date', y='Mud Cutting Ratio', color='Well Name', title='Mud Cutting Ratio'), use_container_width=True)
            st.plotly_chart(px.bar(df, x='Date', y='GPM/Screen', color='Well Name', title='GPM Handled Per Screen'), use_container_width=True)

        with tab4:
            st.subheader("🧠 AI Suggestions")
            if df['LGS%'].mean() > 10 and df['Mud Cutting Ratio'].mean() > 3:
                st.markdown("- 🔺 High LGS% + Cuttings: use finer shaker screens (API 140+).")
            if df['DSRE%'].mean() < 50:
                st.markdown("- ⚠️ Low DSRE%. Check screen seals and alignment.")
            if df['ROP'].mean() < 20:
                st.markdown("- 📉 ROP low. Inspect shaker and solids bypass.")
            if df['PV'].mean() > 35:
                st.markdown("- 💧 High PV: consider mud dilution.")
            if df['Total SCE'].sum() > 400:
                st.markdown("- 🚨 High SCE: evaluate discard routes and treatment.")

        with tab5:
            st.plotly_chart(px.line(df, x='Date', y=['PV', 'YP', 'GPM Total'], title="PV, YP, Total GPM"), use_container_width=True)

        with st.expander("📉 Shaker Screen Wear (Simulated)"):
            df['Screen Wear Index'] = ((df['LGS%'] * df['GPM Total']) / df['API Screen'].replace(0, 100)).clip(upper=150)
            fig = px.line(df, x='Date', y='Screen Wear Index', color='Well Name', title='Shaker Screen Wear Over Time (Estimated)')
            st.plotly_chart(fig, use_container_width=True)
