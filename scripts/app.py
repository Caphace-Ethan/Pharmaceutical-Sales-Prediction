import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px

st.set_page_config(page_title="Dashboard | Telecom User Data Analysis ", layout="wide")

def loadData():
    pd.set_option('max_column', None)
    loaded_data = pd.read_csv('./data/Resp_dataSet.csv')
    return loaded_data

def selectHandset():
    df = loadData()
    handset = st.multiselect("choose Device Type(s)", list(df['device_make'].unique()))
    if handset:
        df = df[np.isin(df, handset).any(axis=1)]
        st.write(df)



st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Telecommunication Users Data Analysis </h1>", unsafe_allow_html=True)
selectHandset()
# st.markdown("<p style='padding:10px; background-color:#000000;color:#00ECB9;font-size:16px;border-radius:10px;'>Section Break</p>", unsafe_allow_html=True)
st.title("Data Visualizations")
# with st.beta_expander("Show More Graphs"):