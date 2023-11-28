# App to predict the volume of traffic
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

st.title('Traffic Volume Prediction: A Machine Learning App')

# display image
st.image('traffic_image.gif', width = 650)

st.subheader("Use this application to predict traffic volume under a variety of conditions")

st.write('Use the form below to get started')

# processing data
traffic_df = pd.read_csv('Traffic_Volume.csv')
traffic_df.head()
traffic_df.replace({np.nan:None}, inplace = True)
traffic_df.drop('weather_description', axis = 1, inplace = True)
traffic_df['date_time']= pd.to_datetime(traffic_df['date_time'],format ='%Y/%m/%d')
traffic_df['month']= traffic_df['date_time'].dt.month
traffic_df['day']= traffic_df['date_time'].dt.day
traffic_df['time'] = pd.to_datetime(traffic_df['date_time']).dt.hour
traffic_df.drop('date_time', axis = 1, inplace = True)

holidays = traffic_df['Holiday'].value_counts()

with st.form(key = 'my form'):
    select_holiday = st.selectbox('Choose what holiday it is', holidays)