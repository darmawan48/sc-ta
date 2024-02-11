import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Page: ', ('EDA', 'Prediksi Opini'))

if navigation == 'EDA':
    eda.start()
else:   
    prediction.start()

