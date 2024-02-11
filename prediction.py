import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from PIL import Image

st.set_page_config(
    page_title='Shopee Customer Opinion Sentiment Predictions',
    layout = 'wide',
    initial_sidebar_state='expanded'
)

# Construct Dataset
data = pd.read_csv('Data ulasan Shopee tentang COD.csv')

# urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
# userPattern = '@[^\s]+'
# def process_opini(opini):
#   # Lower Casing
#     opini = opini.lower()
#     opini=opini[1:]
#     # Removing all URls 
#     opini = re.sub(urlPattern,'',opini)
#     # Removing all @username.
#     opini = re.sub(userPattern,'', opini) 
#     #Remove punctuations
#     opini = opini.translate(str.maketrans("","",string.punctuation))
#     #tokenizing words
#     tokens = word_tokenize(opini)
#     #Removing Stop Words
#     final_tokens = [w for w in tokens if w not in stopword]
#     #reducing a word to its word stem 
#     wordLemm = WordNetLemmatizer()
#     finalwords=[]
#     for w in final_tokens:
#       if len(w)>1:
#         word = wordLemm.lemmatize(w)
#         finalwords.append(word)
#     return ' '.join(finalwords)

# data['processed_opini'] = data['content'].apply(lambda x: process_opini(x))
# print('Text Preprocessing complete.')

#Load All Files
def load_models():
    # Load the vectoriser.
    file = open('vectoriser.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('logisticRegression.pickle', 'rb')
    lg = pickle.load(file)
    file.close()
    return vectoriser, lg

def predict(vectoriser, model, text):
    # Predict the sentiment
    processes_text=[process_opini(sen) for sen in text]
    textdata = vectoriser.transform(processes_text)
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(opini, sentiment):
        data.append((text,pred))
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['opini','sentiment'])
    df = df.replace([0,1,2], ["Negative","Positive","Netral"])
    return df

def start() :
   
       #Membuat title
    st.title('Shopee Customer Opinion Sentiment Predictions')

    #Menambahkan Gambar
    image = Image.open('Opinisentimentp.jpg')
    st.image(image, caption='Shopee Customer Opinion')

    with st.form(key='shopee_predict'):
        textnya = st.text_input('Opini statement Anda', value='')

        submitted = st.form_submit_button('Opini sentiment Prediction')


        # Loading the models.
    vectoriser, lg = load_models()
    
    # Text to classify should be in a list.
    opini = ['textnya']
    
    df = predict(vectoriser, lg, opini)
    st.dataframe(df)

if __name__ == '__main__':
    start()

