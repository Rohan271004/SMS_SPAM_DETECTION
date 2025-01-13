import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Function for text preprocessing
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric tokens
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)


# Load vectorizer and model with error handling
try:
    tk = pickle.load(open("vectorizer.pkl", 'rb'))
    model = pickle.load(open("model.pkl", 'rb'))
except FileNotFoundError:
    st.error("Required files (vectorizer.pkl or model.pkl) not found! Please ensure they are in the same directory as this script.")

# Streamlit App
st.title("📩 SMS Spam Detection Model")
st.subheader("Empowering communication through intelligent technology!")
st.write("### *Developed in collaboration with Edunet Foundation*")

# Add a progress bar for engagement
st.progress(100)

# Input SMS
input_sms = st.text_area("📨 Enter the SMS you want to analyze:")

# Predict Button
if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a valid SMS to analyze.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tk.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("📛 Spam Message")
            st.write("⚠️ **This message might be harmful. Avoid responding or clicking any links.**")
        else:
            st.header("✅ Not Spam")
            st.write("✨ **This message appears safe. Feel free to engage.**")

# Footer with acknowledgment and your name
st.markdown("---")
st.markdown(
    """
    <div style="text-align: right; font-size: 10px; color: gray;">
        <i>Developed by</i> <b>Rohan Y. Panchal</b> <br>
        <i>In collaboration with</i> <b>Edunet Foundation</b>
    </div>
    """, unsafe_allow_html=True
)

# Add a motivational quote to inspire viewers
st.markdown(
    """
    ---
    🌟 *"Innovating technology to create smarter solutions for a better tomorrow!"*
    """)
