import streamlit as st
from st_on_hover_tabs import on_hover_tabs
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
st.set_page_config(layout="wide")

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
@st.cache
def clean_text(text):
    # Remove english stop words
    stop_words = set(stopwords.words('english')) - set(['a','i','o','re','to','an','so','ma'])
    stop_words.update(['compare'])
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

model = tf.keras.models.load_model('improvedmodel.h5')

@st.cache
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

def main():
    st.markdown('<style>' + open('styles.css').read() + '</style>', unsafe_allow_html=True)

    with st.sidebar:
        tabs = on_hover_tabs(tabName=['Home', 'Generate Proverb', 'Contributors'], 
                            iconName=['home', 'grade', 'groups'], default_choice=0)
                              

    if tabs == "Home":
        st.title("Yoruba Proverb Generator App")
        st.subheader('About')
        st.write("""This is a Yoruba proverb generator app. The model takes in a keyword or phrase in Yoruba language and generates a proverb in Yoruba language from the given keyword/phrase. Using the keras tokenizer to generate tokens and a Sequential model by adding an embedding layer and a bidirectional LSTM model. We also took into consideration, homonymns and language intonations to help the model generate a more readable proverb. In addition, using the deep-translator library, we added google translator API to help translate the generated Yoruba proverb to English Language for non-yoruba readers.
        """)

        st.warning('NOTE: The english Translation might not give the accurate translation of the generated Yoruba proverb.')
        st.subheader("How To Use The Web App")
        st.write("""
        1. Input Keyword should take in ONLY Yoruba Keywords/phrases with the right intonation marks.
        2. Number of Generated words is the number of next keywords you want the model to generate for you.
        3. You can only select one of the options under the Number of Generated words which are 5,10,15, or 20.
        4. After performing step 1 & 3, click on the Generate Proverbs button.
        """)
        st.subheader('How To Get The Yoruba Alphabets With Intonation/Marks(Àmì)')
        st.info("""
        TRHOUGH A THIRD PARTY WEBSITE:
        - LEXILOGOS is a multilinguial keyboard. There's a virtual keyboard on the website that you can use in typing the Yoruba Alphabets.
        - Here's the link to the website https://lexilogos.com/keyboard/yoruba.htm

        FOR PC(WINDOWS):
        - Right click on the task bar
        - Click on show touch keyboard
        - Long press the letter on the touch keyboard with the cursor.
        
        FOR MAC OS & PHONES:
        - Make sure you add Yoruba as language preference in your keyword
        - Change the keyboard language to Yoruba and type
        - Voila..... you're good to go
        """)

    elif tabs == 'Generate Proverb':
        st.title("Generate a Yoruba Proverb")
        st.info("""
        Get The Yoruba Alphabets With Intonation/Marks(Àmì):
        - LEXILOGOS is a multilinguial keyboard. There's a virtual keyboard on the website that you can use in typing the Yoruba Alphabets.
        - Here's the link to the website https://lexilogos.com/keyboard/yoruba.htm
        """)
        search_term = st.text_input("Input Keyword")
        next_words = st.selectbox('Number of Generated words',(5,10,15,20))
        if st.button("Generate Proverb"):
            results = generate_proverbs(search_term, next_words)
            st.success('Generated Proverb: {}.'. format(results)) 
            translation = GoogleTranslator(source='auto', target='en').translate(results)
            st.success('English Translation of Generated proverb: {}.'.format(translation))
        

    elif tabs == 'Contributors':
        st.title('Contributors')
        st.text("""
        This app was developed by:
        """)
        st.write('- Dolapo Adebo (https://github.com/aadedolapo)')
        st.write('- Mubar Dauda (https://github.com/mubardauda)')
        st.write('- Oluwatooni Adebiyi (https://github.com/aoluwatooni)')
        st.write('- Samuel Iheagwam (https://github.com/Psalmuel69)')
        st.write('- Ugochukwu (https://github.com/Ugo-1)')
        st.info('During #DSRoom challenge under the mentorship of Samson Afolabi (https://twitter.com/samsonafo)')
if __name__ == '__main__':
    main()
    