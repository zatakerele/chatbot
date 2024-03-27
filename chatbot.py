import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
lemmatizer = nltk.stem.WordNetLemmatizer()

# download required NTLK data 
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


st.markdown("<h1 style = 'color: #5F5D9C; text-align: center; font-family: helvetica'> ORGANIZATIONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive'>Built By ZAtAkerele Data Science Cohort</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)

st.image('pngwing.com(1).png')
st.header('Project Info', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing questions in the FAQ. ")
data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)

cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

df = pd.DataFrame()
df['Question'] = cust[1].reset_index(drop = True)
df['Answer'] = sales[1].reset_index(drop = True)

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)

    preprocess_sentences = []
    for sentence in sentences:
        tokens =[lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]

        preprocessed_sentence = ' '.join(tokens)
        preprocess_sentences.append(preprocessed_sentence)
    return ' '.join(preprocess_sentences)   
df['tokenized Questions'] = df['Question'].apply(preprocess_text)

x = df['tokenized Questions'].to_list()

tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(x)

hist_list = []

robot_image, chat_response = st.columns(2)
with robot_image:
    robot_image.image('pngwing.com (3).png', caption = "Hello, Let's Erase Your Confusion")

with chat_response:
    user_word = chat_response.text_input('Hello User, Ask me Your Questions')


    def get_response(user_input):
        user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

        user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

        similarity_scores = cosine_similarity(user_input_vector, corpus) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

        most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

        return df['Answer'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response

    # create greeting list 
    greetings = ["Hey There.... I am a creation of Ehiz Danny Agba Coder.... How can I help",
                "Hi Human.... How can I help",
                'Twale baba nla, wetin dey happen nah',
                'How far Alaye, wetin happen'
                "Good Day .... How can I help", 
                "Hello There... How can I be useful to you today",
                "Hi GomyCode Student.... How can I be of use"]

    exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
    farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

    random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
    random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list

    # Test your chatbot
    while True:
        user_input = input("You: ")
        if user_input.lower() in exits:
            print(f"\nChatbot: {random_farewell}!")
            break
        if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
            print(f"\nChatbot: {random_greetings}!")
        else:   
            response = get_response(user_input)
            print(f"\nChatbot: {response}")

