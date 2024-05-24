from random import randint
from pydantic import BaseModel
import torch
import re,string
from model.transformer import Seq2SeqTransformer
from utils.utils import Translation
from langdetect import detect
from langdetect import detect_langs
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title='Trang chủ', page_icon="🏠")
st.title('Trang chủ')
st.page_link("pages/2_🤖_Translation.py", label="Translation Page", icon="1️⃣")
st.page_link("pages/3_💩_BLEU.py", label="BLEU Score Page", icon="2️⃣")





