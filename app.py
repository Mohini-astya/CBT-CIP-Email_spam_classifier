
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Cache NLTK data downloads
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

download_nltk_data()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Cache loading of vectorizer and model
@st.cache_resource
def load_vectorizer():
    return pickle.load(open('vectorizer.pkl', 'rb'))

@st.cache_resource
def load_model():
    return pickle.load(open('model1.pkl', 'rb'))

# Load vectorizer and model
tfidf = load_vectorizer()
model = load_model()
print("Model loaded:", model)
if hasattr(model, "classes_"):
    print("Model is fitted.")
else:
    print("Model is not fitted.")


# Streamlit app interface
st.title('Email Spam Classifier')

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():
        # Preprocess, vectorize, and predict
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display the result
        st.header("Spam" if result == 1 else "Not Spam")
    else:
        st.error("Please enter a valid message!")
