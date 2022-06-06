import re
import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences




# Function to pre-process text
def preprocess(phrase): 
  
    phrase = phrase.lower()   
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub('[^\w\s]','', phrase).strip()

    return phrase



# loading tokenizer object
with open('tokenizer.pkl', 'rb') as f:
    t = pickle.load(f)


# loading best model
model = load_model('bi_lstm_model.h5')



def predict(s):
    '''This function takes a comment(string) as input and 
       returns whether the type emotions'''
    
    # Convert input string to list
    inp_str = [preprocess(s)]

    # Tokenize input string
    encoded_str = t.texts_to_sequences(inp_str)

    # Padding input sequence to have length of 30
    padded_str = pad_sequences(encoded_str, maxlen=300, dtype='int32', 
                               padding='post', truncating='post', value=0.0)
    
    # prediction on padded input sequence
    predict_y = model.predict(padded_str) # predict using model
    classes_y = np.argmax(predict_y,axis=1) # getting class labels


    # Output string
    if classes_y == 0:
        op_str = 'The above emotion is of Joy' 

    elif classes_y == 1:
        op_str = 'The above emotion is of Anger'

    elif classes_y == 2:
        op_str = 'The above emotion is of Love'

    elif classes_y == 3:
        op_str = 'The above emotion is of Sadness'
    
    elif classes_y == 4:
        op_str = 'The above emotion is of Fear'

    else:
        op_str = 'The above emotion  is of Surprise'
    
    return op_str



def main():
    st.set_page_config(page_title="EMOTION DETECTION", 
                       page_icon=":robot_face:",
                       layout="wide",
                       )
    st.markdown("<h4 style='text-align: center; color:grey;'>Emotions Detection with NLP &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">Detect Emotions</h3>', unsafe_allow_html=True)
    st.text('')
    input_text = st.text_area("Enter text and click on Predict to know if the emotions", max_chars=500, height=150)
    if st.button("Predict"):
        output = predict(input_text)
        st.write(output)
 
if __name__=='__main__':
    main()