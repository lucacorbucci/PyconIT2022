import streamlit as st
from utils import *

st.title("Mnist Example")

model = load_model()


uploaded_file = st.file_uploader("Upload the image you want to classify:")
st.write("## Let's see the image:")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    prediction = make_prediction(model, image)
    st.write(prediction)