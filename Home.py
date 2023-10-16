import streamlit as st

st.set_page_config(
    page_title="Weather Data Dashboard",
    page_icon="🌤️",
)

        
st.write("# Welcome to the Weather Data Dashboard! 🌤️")
#st.sidebar.success("Home")

st.markdown(
    """
    The Weather Data Dashboard is a compound of techniques to build and deploy dashboards apps using Streamlit library and hosting services.
    I have decide to use two pages to display:
    
      1. Local: Based on a dataset containing weather observations recorded in one unknown local.
      2. National: Based on INMET 2022 data available at https://portal.inmet.gov.br/, including geodata information.
    
    Just click in one of the options on the left.
    
    Code available in https://github.com/hcordioli/weather
    
    I hope you enjoy.
    
"""
)