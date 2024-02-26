import streamlit as st
import dataprocessing as db
import requests
movie=st.selectbox('Enter your movie choice',db.m_list())
btn=st.button('Recommend')
if btn:
   recc= db.recommend(movie)
   for i in recc:
       st.write(i)
       