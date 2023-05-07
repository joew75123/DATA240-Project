from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from sklearn.preprocessing import normalize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Define function to perform text preprocessing
def preprocess_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Tokenization
    tokens = word_tokenize(text)
    # Stopwords removing
    stopwords_english = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords_english]
    # Convert to lower case
    text = text.lower()
    # Lemmatization/stemming
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)



class FeatureExtraction:
    def __init__(self):
        pass
    
    def feature_description_col(self,df):
        df['description'] = df['description'].apply(preprocess_text)
        description = [word_tokenize(d) for d in df['description']]
        w2v_sg = Word2Vec(description, vector_size=100, window=5, min_count=1, sg=1)  # set sg=1 for skip-gram
        def get_w2v_sg_features(description):
            words = word_tokenize(description)
            feature_vec = np.zeros((100,), dtype="float32")
            for word in words:
                if word in w2v_sg.wv.key_to_index:
                    context_words = w2v_sg.wv.most_similar(word, topn=5)  # get top 5 similar words from skip-gram model
                    for context_word, _ in context_words:
                        feature_vec = np.add(feature_vec, w2v_sg.wv[context_word])
                else:
                    pass
            feature_vec = normalize([feature_vec])[0]
            return feature_vec
        d = [get_w2v_sg_features(d) for d in df['description']]
        d = pd.DataFrame(d, columns=['description_word2vec{}'.format(i) for i in range(1, 101)])
        df = pd.merge(df, d, left_index=True, right_index=True)
        return df