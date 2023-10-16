import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib.colors import to_hex
import plotly.express as px
import matplotlib.pyplot as plt
import branca 
from branca.colormap import linear
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point, Polygon
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math
from streamlit_folium import folium_static
import datetime as dt
from datetime import timedelta
from datetime import datetime

# Set layout
st.set_page_config(layout="wide")

# Maps
map_width=500
map_height=500
br_lat_center = -15
br_lon_center = -55
UFs = ("AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO")

# Temperature indicators
TEMP_INDICATORS = {
    "mean":
        [
            "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean",
            "TEMPERATURA DO PONTO DE ORVALHO (°C)_mean",
            "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_mean",
            "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)_mean",
            "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)_mean"
        ],
    "min":
        [
            "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)_min",
            "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)_min"
        ],
    "max":
        [
            "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max",
            "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)_max"
        ]   
}

# Ploting functions
mmm_variables = {
    "Umidade":"UMIDADE RELATIVA DO AR, HORARIA (%)_",
    "Pressão":"PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)_",
    "Temperatura":"TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_",
    "Vento":"VENTO, VELOCIDADE HORARIA (m/s)_"
}

# Ajuste dos dados
@st.cache_data
def load_inmet_data():
    station_obs=pd.read_csv('./data/inmet_2022_daily_observations.csv',index_col="Data")
    station_params = pd.read_csv('./data/inmet_2022_stations.csv')
    inmet_df = pd.merge(station_obs.reset_index(), station_params, how='left', on='CODIGO (WMO)').set_index('Data')
    inmet_df.drop("DATA DE FUNDACAO",axis = 1, inplace=True)
    inmet_df.index = pd.to_datetime(inmet_df.index)
    
    # All supporting info structures for the Poligono das Secas Map
    ps_border = ["JEREMOABO","CURACA","SAO JOAO DO PIAUI","CANTO DO BURITI","ALVORADA DO GURGUEIA","BALSAS","DIANOPOLIS","POSSE","CHAPADA GAUCHA","MONTALVANIA","VITORIA DA CONQUISTA","ITIRUCU","ITABERABA","FEIRA DE SANTANA","RIBEIRA DO AMPARO"]
    temp_df = station_params.loc[station_params['ESTACAO'].isin(ps_border)]
    ps_df = pd.DataFrame()
    for stat in ps_border:
        row = temp_df.loc[temp_df["ESTACAO"] == stat]
        ps_df = pd.concat([ps_df,row])

    ps_codes = ps_df['CODIGO (WMO)'].values
    ps_data = station_obs.loc[station_obs['CODIGO (WMO)'].isin(ps_codes)]
    ps_max_temps = ps_data.groupby("CODIGO (WMO)")[["TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max"]].agg("max")

    lats = ps_df['LATITUDE'].values
    longs = ps_df['LONGITUDE'].values
    stations = ps_df['CODIGO (WMO)'].values
    ps_points = [ Point(lats[i],longs[i]) for i in range(0,len(lats))]
    ps_polygon = Polygon(zip(longs, lats))
    ps_gpd = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[ps_polygon])
    tm = ps_max_temps['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max'].mean()
    t_df = pd.DataFrame(columns=['CODIGO (WMO)','TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max'])
    for index, row in ps_df.iterrows():
        row_df = pd.DataFrame([[row['CODIGO (WMO)'], tm]],columns=['CODIGO (WMO)','TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max'])
        t_df = pd.concat([t_df,row_df])
    t_df.set_index("CODIGO (WMO)",drop=True,inplace=True)

    # Info structures for Min, Mean and Max plot 
    station_obs['temp_variation'] = station_obs["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_max"] - station_obs["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_min"]
    highest_max_min = station_obs.sort_values("temp_variation",ascending=False)[[
        "CODIGO (WMO)","temp_variation","TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_max","TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_min"
    ]]
    highest_max_min = pd.merge(highest_max_min.reset_index(), station_params, how='inner', on='CODIGO (WMO)').set_index('Data')   
    hmm = highest_max_min.groupby([highest_max_min.index,"ESTACAO"])[["temp_variation","UF"]].agg("max").reset_index().set_index('Data').sort_values("temp_variation",ascending=False)
    cidades_maior_variacao = list(set(hmm["ESTACAO"].head(10).values))
    cidades_menor_variacao = list(set(hmm["ESTACAO"].tail(10).values))
    mv_df = station_params.loc[station_params['ESTACAO'].isin(cidades_maior_variacao)].reset_index()
    temp_min = station_obs.groupby("CODIGO (WMO)").agg("min")[TEMP_INDICATORS['min']]
    temp_min = temp_min.sort_values("TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)_min",ascending=True).head(10)
    temp_min = pd.merge(temp_min,station_params,how='left',on="CODIGO (WMO)")  
    temp_max = station_obs.groupby("CODIGO (WMO)").agg("max")[TEMP_INDICATORS['max']]
    temp_max = temp_max.sort_values("TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max",ascending=False).head(10)
    temp_max = pd.merge(temp_max,station_params,how='left',on="CODIGO (WMO)")

    return(inmet_df,station_params,station_obs,ps_df,ps_gpd,t_df,ps_df,mv_df,temp_max,temp_min)

# Create a text element and let the reader know the data is loading.
inmet_df,station_params,station_obs,ps_df,ps_gpd,t_df,ps_df,mv_df,temp_max,temp_min = load_inmet_data()

# Divide a tela do Dashboard em espaços para os gráficos
col1,col2 = st.columns((3,3))
col3,col4 = st.columns([0.95, 0.05])
col5,col6 = st.columns([0.95, 0.05])
col6,col7 = st.columns([0.95, 0.05])

with col1:
    br = folium.Map(location=[br_lat_center, br_lon_center], zoom_start=4, control_scale=True)
    marker_cluster = MarkerCluster().add_to(br)

    #add a marker for each station, add it to the cluster, not the map
    for index, row in station_params.iterrows():
        folium.Marker(
            location=[row['LATITUDE'],row['LONGITUDE']],
            popup="Add popup text here.",
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(marker_cluster)

    folium_static(br,width=map_width, height=map_height)
    
with col2:
    year_avg_temp = inmet_df.groupby("CODIGO (WMO)")[[("TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean")]].mean()
    year_avg_temp = pd.merge(year_avg_temp, station_params, how='left', on='CODIGO (WMO)')
    year_avg_temp = year_avg_temp[year_avg_temp["TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean"].notna()]
    
    m = folium.Map([br_lat_center, br_lon_center], zoom_start=4)

    #specify the min and max values of your data
    colormap = branca.colormap.linear.Set1_05.scale(0, 10)
    colormap = colormap.to_step(index=[5,15, 20, 27, 30, 40])
    colormap.caption = 'Average Temperatures in INMET stations'
    colormap.add_to(m)
    out = list(map(to_hex, colormap.colors))

    for i in range(0,len(year_avg_temp)):
        if (year_avg_temp.iloc[i]['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean']) <= 5:
            station_color =  out[0]
        elif year_avg_temp.iloc[i]['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean'] <= 15:
            station_color =  out[1]
        elif year_avg_temp.iloc[i]['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean'] <= 20:
            station_color =  out[2]       
        elif year_avg_temp.iloc[i]['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean'] <= 27:
            station_color =  out[3]
        else:
            station_color =  out[4]
        
        folium.Marker(
            location=[year_avg_temp.iloc[i]['LATITUDE'], year_avg_temp.iloc[i]['LONGITUDE']],
            popup=round(year_avg_temp.iloc[i]['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean'],1),
            #icon=folium.DivIcon(html=f"""<div style="font-family: courier new; color: {station_color}">{year_avg_temp.iloc[i]['temp_str']}</div>""")
            icon=folium.Icon(color = "white", icon_color=station_color),
        ).add_to(m) 
    folium_static(m,width=map_width, height=map_height)
    
with col3:
    m_states = folium.Map(location=(-15, -48), zoom_start=4, tiles="cartodb positron")

    folium.Choropleth(
        geo_data="./data/Brazil.json",
        data=inmet_df,
        columns=["UF", "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)_mean"],
        key_on="feature.properties.UF",
        fill_color="RdYlGn_r",
        fill_opacity=0.8,
        line_opacity=0.3,
        nan_fill_color="white",
    ).add_to(m_states)

    # Mark the locations of PS
    for index, row in ps_df.iterrows():
        popup_text = row['ESTACAO'] + ": " + str(t_df.loc[row['CODIGO (WMO)']][0])
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=popup_text,
            icon=folium.Icon(color="red"),
        ).add_to(m_states)

    # The PS Polygon
    sim_geo = gpd.GeoSeries(ps_gpd["geometry"]).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function=lambda x: {"fillColor": "darkred"})
    geo_j.add_to(m_states)

    # Again, the stations with highest vairance in temperatures
    for index, row in mv_df.iterrows():
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=row['ESTACAO'],
            icon=folium.Icon(color="darkred"),
        ).add_to(m_states)

    # Mark the top 10 locations of highest temps
    for index, row in temp_max.iterrows():    
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=row['TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)_max'],
            icon=folium.Icon(color="red"),
        ).add_to(m_states)

    # Mark the top 10 locations of lowest temps
    for index, row in temp_min.iterrows():    
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=row['TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)_min'],
            icon=folium.Icon(color="green"),
        ).add_to(m_states)
    folium_static(m_states,width=map_width*2, height=map_height)

with col5:
    # Plot Min, Mean and Max variable
    def plot_mmm(station):
        mmm_df = inmet_df.loc[inmet_df['ESTACAO'] == station] 
        fig = make_subplots(rows=4, cols=1,shared_xaxes=True,vertical_spacing=0.03,
                        subplot_titles=([key for key in mmm_variables.keys()]))
        
        x_axis=[d.strftime("%m/%d") for d in mmm_df.index]
        for idx, var in enumerate(mmm_variables):
            mmm_cols = [mmm_variables[var] + type for type in ["mean","min","max"]]
            fig.add_scatter(x=x_axis, y=mmm_df[mmm_cols[0]], mode='lines',name='Mean',line_width=1,row=idx+1, col=1)
            fig.add_scatter(x=x_axis, y=mmm_df[mmm_cols[1]], mode='lines',name='Max',line_width=1,row=idx+1, col=1)
            fig.add_scatter(x=x_axis, y=mmm_df[mmm_cols[2]], mode='lines',name="Min",line_width=1,row=idx+1, col=1)
            
        chart_label = "Médias, Máximas e Mínimas em " + station
        fig.update_layout(
            height=800, width=980, title_text=chart_label,showlegend=False
        )
        fig.update_annotations(font_size=10)
        return(fig)
    
    # Stations by UF
    def get_stations_by_uf(uf):
        return (sorted(list(station_params.loc[station_params['UF'] == uf]['ESTACAO'].values)))

    uf_option = st.selectbox(
        'Select UF',
        UFs,index=UFs.index("SP")
    )
    stations_by_uf = get_stations_by_uf(uf_option)
    station_option = st.selectbox(
        'Select Station',
        stations_by_uf
    )
    col5.plotly_chart(plot_mmm(station_option),use_container_width=True)