import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    pv_match = re.search(r"PV\s*@\s*°F\s*[:=]?\s*([0-9.]+)", text)
    data['PV'] = pv_match.group(1) if pv_match else '0'
    yp_match = re.search(r"YP\s+lb/100ft²\s*([0-9.]+)", text)
    data['YP'] = yp_match.group(1) if yp_match else '0'
    mw_match = re.search(r"Density\s+@\s*°F\s*([0-9.]+)", text)
    data['Mud Weight'] = mw_match.group(1) if mw_match else '0'
    temp_match = re.search(r"Fl Temp\s*[°F]*\s*([0-9.]+)", text)
    data['Ave Temp'] = temp_match.group(1) if temp_match else '0'
    gpm_match = re.findall(r"GPM\s+([0-9.]+)", text)
    data['Pump 1 GPM'] = gpm_match[2] if len(gpm_match) > 2 else '0'
    data['Pump 2 GPM'] = gpm_match[0] if len(gpm_match) > 0 else '0'
    data['Pump 3 GPM'] = gpm_match[1] if len(gpm_match) > 1 else '0'
    screen_lines = re.findall(r"(NOV|Derrick).*?(\d+\s+\d+\s+\d+\s+\d+)", text)
    api_screens = [int(x) for _, block in screen_lines for x in re.findall(r"\d+", block)]
    data['API Screen'] = str(sum(api_screens) / len(api_screens)) if api_screens else '0'
    data['Screen Count'] = str(len(api_screens))
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

        with st.sidebar:
            st.header("🔍 Well Filter")
            well_options = df['Well Name'].dropna().unique().tolist()
            selected_wells = st.multiselect("Select Well(s)", well_options, default=well_options)
            df = df[df['Well Name'].isin(selected_wells)]
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
        df['API Screen'] = df['API Screen'].apply(to_float)
        df['GPM/Screen'] = df['GPM Total'] / df['Screen Count'].replace(0, 1)
        df['Dilution vs SCE Ratio'] = df['Total Dilution'] / df['Total SCE'].replace(0, 1)
        df['Top Deck Wear'] = (df['GPM Total'] * df['LGS%']) / df['API Screen'].replace(0, 100)
        df['Bottom Deck Wear'] = (df['GPM/Screen'] * 0.8).clip(upper=100)
        df['Dilution vs SCE Ratio'] = df['Total Dilution'] / df['Total SCE'].replace(0, 1)
        df['API Screen'] = df['API Screen'].apply(to_float)

        st.dataframe(df)

        st.subheader("📋 Daily Summary Report by Well")
        summary = df.groupby(['Date', 'Well Name']).agg({
            'Total Circ': 'sum',
            'LGS%': 'mean',
            'DSRE%': 'mean',
            'Total SCE': 'sum',
            'ROP': 'mean',
            'Top Deck Wear': 'mean',
            'Bottom Deck Wear': 'mean'
        }).round(2).reset_index()
        st.dataframe(summary)

        st.subheader("📌 Wear Benchmark Flags")
        flagged = df[['Date', 'Well Name', 'Top Deck Wear', 'Bottom Deck Wear', 'API Screen', 'Screen Count']]
        flagged['Top Deck Status'] = pd.cut(flagged['Top Deck Wear'], bins=[-1, 2.5, 4.0, 999], labels=["✅ Good", "⚠️ Caution", "🚨 Critical"])
        flagged['Bottom Deck Status'] = pd.cut(flagged['Bottom Deck Wear'], bins=[-1, 2.0, 3.5, 999], labels=["✅ Good", "⚠️ Caution", "🚨 Critical"])
        st.dataframe(flagged)

        st.subheader("📊 Statistical Summary (Key Metrics)")
        st.dataframe(df[['ROP', 'DSRE%', 'LGS%', 'Top Deck Wear', 'Dilution vs SCE Ratio']].describe().T[['mean', 'std', 'min', 'max']].round(2))

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
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df['Date'], y=df['GPM Total'], name='GPM Total', marker_color='indianred'))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['PV'], name='PV', mode='lines+markers', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['YP'], name='YP', mode='lines+markers', line=dict(color='green')))
            fig.update_layout(title='PV, YP and GPM Total (Combo View)', yaxis_title='Value', barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
