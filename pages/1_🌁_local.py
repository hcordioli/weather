import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
import numpy as np
import plotly.graph_objects as go

# Set layout
st.set_page_config(layout="wide")

# Global Variables
variable_options = (
    'Temperature (C)','Apparent Temperature (C)','Humidity','Wind Speed (km/h)',
    'Wind Bearing (degrees)','Visibility (km)','Pressure (millibars)'  
)


# Ajuste dos dados
@st.cache_data
def load_data():
    df_weather = pd.read_csv("./data/weatherHistory.csv")
    df_weather.drop(columns=["Loud Cover"],inplace=True)
    def setPreciptationType(ptype, temp, hum):
        if pd.isna(ptype) and temp < 0 and hum > 0.8:
            myvalue="snow"
        elif pd.isna(ptype):
            myvalue="rain"
        else:
            myvalue=ptype
        return myvalue
    df_weather['Precip Type'] = df_weather.apply(lambda x: setPreciptationType(x['Precip Type'], x['Apparent Temperature (C)'], x['Humidity']), axis=1)
    df_weather['Formatted Date']=pd.to_datetime(df_weather['Formatted Date'], utc=True).dt.date
    df_weather['Formatted Date'] = pd.to_datetime(df_weather['Formatted Date'])
    df_weather['year'] = df_weather['Formatted Date'].dt.year
    df_weather['month'] = df_weather['Formatted Date'].dt.month
    df_weather = df_weather.drop(df_weather.loc[df_weather['year']==2005].index)
    return(df_weather)

def filter_df(df,year_option):
    # First, filter by year
    if year_option != "all":
        year_option_int = int(year_option)
        df_filtered = df.loc[df["year"] == year_option_int].select_dtypes(include='number')
    else:
        df_filtered = df.select_dtypes(include='number')
    
    # Group by month
    df_filtered = df_filtered.groupby("month").agg(["min","mean","max"])
    
    return (df_filtered)

def std(x): 
    return np.std(x)

# Create a text element and let the reader know the data is loading.
df_weather = load_data()

# Divide a tela do Dashboard em espaços para os gráficos
col1,col2 = st.columns([0.9,0.1])
col3,col4 = st.columns([1, 3])
col5,col6 = st.columns([1, 3])
col7,col8 = st.columns([1, 3])

with col1:
    correl_df = df_weather.select_dtypes(include='number').corr().round(decimals=1)
    fig_correl = px.imshow(correl_df,text_auto=True,
                           width=1000, height=600,title="Weather variables correlation"
                           )
    col1.plotly_chart(fig_correl,use_container_width=False)

with col3:
    selected_weather_var = st.selectbox(
        'Select weather variable',
        variable_options)
    
with col4:
    df_summary = df_weather.groupby(["Summary","Precip Type"])[selected_weather_var].agg(["mean",std]).reset_index()
    # assign colors to type using a dictionary
    colors = {'snow':'steelblue',
            'rain':'firebrick'}
    color_map = [colors[t] for t in df_summary["Precip Type"]]

    fig_sum = go.Figure()
    fig_sum.add_trace(
        go.Bar(
            name='Temp',
            x=df_summary["Summary"], 
            y=df_summary["mean"],
            marker_color=color_map,
            error_y=dict(type='data', array=df_summary["std"])
    ))
    col4.plotly_chart(fig_sum,use_container_width=True)
    
    
with col5:
    selected_weather_var_2 = st.selectbox(
        'Select weather variable_',
        variable_options)
    select_year = st.selectbox(
        'Select one year or all',
        list(df_weather.year.value_counts().sort_index().index.values)+["all"]
    )

with col6:
    df_filtered = filter_df(df_weather,select_year)
    picked_columns = [(selected_weather_var_2,"min"),(selected_weather_var_2,"mean"),(selected_weather_var_2,"max")]
    df_display = df_filtered[picked_columns].reset_index(col_level=1)
    df_display.columns = df_display.columns.get_level_values(1)
    df_display = pd.melt(df_display, id_vars=['month'], value_vars=['min', 'mean','max'])
    fig_month = px.bar(df_display, x='month', y='value', color='variable' ,title="Monthly "+selected_weather_var_2)
    col6.plotly_chart(fig_month,use_container_width=True)

with col7:
    selected_weather_var_3 = st.selectbox(
        'Select weather variable__',
        variable_options)
    select_year_ = st.selectbox(
        'Select one year ',
        list(df_weather.year.value_counts().sort_index().index.values)
    ) 

with col8:
    df_year = df_weather.loc[df_weather['year']==select_year_].sort_values("Formatted Date")[["Formatted Date",selected_weather_var_3]]
    picked_columns = [(selected_weather_var_3,"min"),(selected_weather_var_3,"mean"),(selected_weather_var_3,"max")]
    df_year = df_year.groupby("Formatted Date")[selected_weather_var_3].agg(["min","mean","max"]).reset_index()
    df_display = pd.melt(df_year, id_vars=['Formatted Date'], value_vars=['min', 'mean','max'])
    fig_year = px.line(df_display, x='Formatted Date', y='value', color='variable' ,title="Daily Averages "+selected_weather_var_3)
    col8.plotly_chart(fig_year,use_container_width=True)
