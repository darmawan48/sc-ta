import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title='Shopee Customer Opinion Sentiment Predictions',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

def start():

    #Membuat title
    st.title('Shopee Customer Opinion Sentiment Predictions')

    #Membuat Sub Header     
    st. subheader('EDA untuk Shopee Customer Opinion Sentiment Predictions')

    #Menambahkan Gambar
    image = Image.open('Opinisentiment.jpg')
    st.image(image, caption='Shopee Customer Opinion')
    #Menambahkan Deskripsi
    st.write('Tugas Akhir Sanber Campus ITB  **Risanto.Darmawan**')
    st.write('# Batch 1')
    #Membuat Garis lurus
    st.markdown('---')

    #Magic Syntax
    '''
    Page ini merupakan explorasi sederhana opini Shopee Customer
    Dataset yang digunakan adalah dari Kaggle.com
    Dataset ini berasal dari https://www.kaggle.com/code/alvianardiansyah/analisis-sentimen-pengguna-shopee-dengan-svm/input
    '''

    #Show Dataframe
    data = pd.read_csv('Data ulasan Shopee tentang COD.csv')
    st.dataframe(data)

    #Membuat Barplot
    st.write('### Plot Score Opini Customer Shopee mengenai COD')
    st.write('Score 1 = Tidak Baik, 2 = Tidak cukup baik. 3 = cukup, 4 = Baik, 5 = Baik Sekali')
    fig = plt.figure(figsize=(15, 5))
    sns.countplot(x='score', data=data)
    st.pyplot(fig)

    
    data['score'] = data['score'].replace(1,0)
    data['score'] = data['score'].replace(2,0)

    data['score'] = data['score'].replace(4,1)
    data['score'] = data['score'].replace(5,1)

    data['score'] = data['score'].replace(3,2)


    positif = data['score'][data.score == 1 ]
    negatif = data['score'][data.score == 0 ]
    netral = data['score'][data.score == 2 ]

    st.write('### Merging Score')
    st.write('Score 0 = Negative, 1 = Positive, 2 = Netral')

    fig = plt.figure(figsize=(15, 5))
    sns.countplot(x='score', data=data)
    st.pyplot(fig)




if __name__ == '__main__':
    start()
