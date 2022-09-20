import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import re
import nltk
from deep_translator import GoogleTranslator

# Load data
with open("owe.txt", "r", encoding='utf-8') as file:
  data = file.readlines()
  # using regular expression to remove punctuations from each line
  data = re.sub(r"[^\w\s]","",''.join(data)).lower().split('\n')
  # data = ' '.join(data).lower().split('\n')
del data[0]
del data[-1]
del data[50]
del data[823]
del data[2075:2077]
del data[2076:2079]

df = pd.DataFrame(data, columns=['owe'])
df.head()

nltk.download('stopwords')
from nltk.corpus import stopwords
# Function for text cleaning
def clean_text(text):
    # Remove english stop words
    stop_words = set(stopwords.words('english')) - set(['a','i','o','re','to','an','so','ma'])
    cleaned_text = ' '.join([w for w in text.split() if  not w in stop_words])
    return cleaned_text
df['clean_owe'] = df['owe'].apply(clean_text)
# convert column to a list of each row
clean_data = [line for line in df['clean_owe']]

tokenizer = Tokenizer()
corpus = clean_data
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len,padding='pre'))

model = tf.keras.models.load_model('model.h5')

def generate_proverbs(seed_text,next_words):
    """ A function that takes a 
    seed_text: to prompt next word prediction
    next_word: The number of next words to predict
    """
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        classes=np.argmax(predicted,axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == classes:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


# @st.cache
# def search_term_if_not_found(search_term,df):
# 	result_df = df[df['owe'].str.contains(search_term)]
# 	return result_df['owe'].drop_duplicates()

def main():
    # st.title("Yoruba Proverbs Generator App")
   
    html_temp = """
    <div style="background-color:#f63366;
        border-radius: 25px;
        padding:5px">
    <h2 style="color:white;
        text-align:center;">Yoruba Proverbs Generator App</h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.write('This app takes in a keyword in yoruba language and generates a proverb')
    st.subheader("Generate a Yoruba Proverb")
    search_term = st.text_input("Input Keyword")
    next_words = st.selectbox('Number of Generated words',(5,10,15,20))
    if st.button("Generate Proverb"):
        results = generate_proverbs(search_term, next_words)
        st.success('Generated Proverb: {}.'. format(results)) 
        translation = GoogleTranslator(source='auto', target='en').translate(results)
        st.success('English Translation of Generated proverb: {}.'.format(translation))


    if st.button('About'):
        st.text("Built by Dolapo, Tooni, Samuel, Mubar, and Ugochukwu")
        
if __name__ == '__main__':
    main()
    