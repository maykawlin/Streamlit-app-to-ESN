import library.New_ESN
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import datetime

st.write("""
You must choose the options bellow to train and forecast with our neural network.
""")

option = st.selectbox(
    'Do you want upload data or use Yahoo Finance?',
    ('Select','Upload', 'Yahoo Finance'))

if option == 'Upload':
    uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
    if uploaded_file:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)
elif option == 'Yahoo Finance':
    st.write("""
    
    """)


    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    col1, col2 = st.columns(2)

    with col1:
        stock_code = st.text_input('Stock Code', 'Ex.: PETR4')
        if stock_code != 'Ex.: PETR4' and stock_code != '':
            initial_date = st.date_input("Put the initial date:",datetime.date(2019, 7, 6))
            if initial_date != datetime.date(2019, 7, 6):
                final_date = st.date_input("Put the final date:",datetime.date(2019, 7, 6))
    with col2:
        pass
        # option = st.selectbox(
        #     "How would you like to be contacted?",
        #     ("Email", "Home phone", "Mobile phone"),
        #     label_visibility=st.session_state.visibility,
        #     disabled=st.session_state.disabled,
        # )
