import streamlit as st
import json
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="📈Homepage"
)

with open("pages\styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

col1,col2=st.columns(2)

with col2:
    lottie_stocks=load_lottiefile("pages\stocks.json")  # replace link to local lottie file
    st_lottie(lottie_stocks,height=300,width=400)
with col1:
    st.title('Welcome to StockSmart!')
    st.write('\n\n\n')
    st.write("Presenting StockSmart - The Golden Path to Stock Prognostication.We are here to assist you every step of the way, providing invaluable insights and empowering you to make informed decisions. With our cutting-edge tools and comprehensive resources, we enable you to navigate the dynamic stock market landscape with confidence. Unleash the power of prediction as we unravel the opening range of stock prices, granting you a glimpse into the exciting realm of market trends.")