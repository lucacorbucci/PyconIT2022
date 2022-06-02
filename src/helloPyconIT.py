import streamlit as st
import pandas as pd
import time 

st.title("Hello Pycon IT")

@st.cache
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    time.sleep(2)
    return data

df = load_data("./race-winners.csv")
st.dataframe(df)

class_type = st.radio(
     "Choose the class you want to display",
     ('Moto3™', 'Moto2™', 'MotoGP™'))


df1 = df[df['Class'] == class_type]
count = df1['Rider'].value_counts().head(20)
st.bar_chart(count)

option = st.selectbox(
     'Select the year you want to display:',
     (i for i in range(1949, 2023)))

df1 = df1[df1['Season'] == option]
count = df1['Rider'].value_counts().head(20)
st.bar_chart(count)