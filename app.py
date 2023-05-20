import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


pstem = PorterStemmer()
v = TfidfVectorizer()



vector_form = pickle.load(open('venv/vector.pkl', 'rb'))
load_model = pickle.load(open('venv/model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[pstem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


#[theme] # You have to add this line

primaryColor = '#FF8C02' # Bright Orange

backgroundColor = '#00325B' # Dark Blue

secondaryBackgroundColor = '#55B2FF' # Lighter Blue

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)



if __name__ == '__main__':
    st.title('Fake News Detection Site ')
    st.subheader("Input the News below to know which is fake and which is right")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict the news")
    if predict_btt:
        pred=fake_news(sentence)
        print(pred)
        if pred == [0]:
            st.success('Real')
        if pred == [1]:
            st.warning('Fake News! Please dont get manipulate by this news/n')

st.write(" ")
st.write("Our Github Profiles....")
st.write("(<https://gitHub.com/Zedoman>)")
st.write("(<https://gitHub.com/shenmareparas>)")
st.write("(<https://gitHub.com/shivani2190>)")
st.write("(<https://gitHub.com/shiv12a10>)")
st.write("(<https://gitHub.com/anuraanpaul>)")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

