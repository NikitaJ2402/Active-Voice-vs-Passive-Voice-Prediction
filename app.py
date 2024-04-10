import streamlit as st
import pickle
import numpy as np 
import pandas as pd

st.set_page_config(page_title="Active vs Passive Voice Prediction" )
st.title('A web Application Deployment')

# ##################################################################################################

data = pd.read_excel("D:\Imverse AI assignment\immverse_ai_eval_dataset.xlsx")
st.write('Shape of Dataset :',data.shape)
menu=st.sidebar.radio('Menu',['home','Prediction of Statements'])

if menu=='home':
    st.header('Tabular Data of 40 Top Statements to Understand Active and Passive Voice')
    if st.checkbox('Tabular Data'):
        st.table(data.head(40))

    st.header('Statistical Summary of a DataFrame')
    if st.checkbox('Statistics'):
        st.table(data.describe())


vect = pickle.load(open('vectorizer.sav', 'rb'))
model = pickle.load(open('model.sav', 'rb'))



def prediction(text):
          text_v = vect.transform([text])
          result = model.predict(text_v)

          if result == 1:
              return 'This is a Passive Statement'
          else:
              return 'This is an Active Statement'

if menu=='Prediction of Statements':
        def prediction(text):
          text_v = vect.transform([text])
          result = model.predict(text_v)

          if result == 1:
              return 'This is a Passive Statement'
          else:
              return 'This is an Active Statement'

st.write('')
st.write('')
st.markdown('#### Text Classification for Active vs. Passive Voice Detection')


st.write('')
st.write('')
txt = st.text_area('Write Your Sentence to check whether Active or Passive Voice : ')


st.write('')
st.write('')
st.write('')
btn = st.button('Predict')

st.write('')
st.write('')
st.write('')

if btn:
    p_result = prediction(txt)
    st.markdown(f'# {p_result}')




        



st.write('')
st.write('')
st.write('')
st.write('')
st.caption("A Demo Model :")
st.markdown('<span>Created By <a href="https://www.linkedin.com/in/nikita-jiwani-b816a0212/">Nikita Jiwani</a></span>',unsafe_allow_html=True)