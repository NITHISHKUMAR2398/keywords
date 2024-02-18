# Install the datasets from the hugging face
# pip install datasets
# Import Libraries

import streamlit as st
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from datasets import load_dataset
from utils.utils import download_nltk_data, preprocess_text, KeywordExtractor

# Load the dataset
dataset = load_dataset("ag_news", split='test')
df = pd.DataFrame(dataset)

# Group the dataset containing 250 data points from each class
grouped_data = df.groupby('label')
sampled_data = pd.DataFrame()
num_samples_per_class = 250
for name, group in grouped_data:
    sampled_data = pd.concat([sampled_data, group.sample(n=num_samples_per_class, random_state=42)])

sampled_data.reset_index(drop=True, inplace=True)

# Shuffle the data
shuffled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
data = pd.DataFrame(shuffled_data)

class DataPreprocessor:
    """
    Class for preprocessing text data.

    This class handles tasks like downloading necessary NLTK data, removing stopwords,
    and lemmatizing words.

    Attributes:
        stop_words (list): List of stop words for English language.
        lmtzr (WordNetLemmatizer): Instance of WordNetLemmatizer for lemmatizing words.

    Methods:
        pre_process(text): Preprocesses the input text by removing stop words and lemmatizing.
    """
    def __init__(self):
        """
        Initializes DataPreprocessor with stop words and WordNetLemmatizer.
        """
        download_nltk_data()
        stop_words = set(stopwords.words('english'))
        new_words = ["fig", "figure", "image", "sample", "using",
                     "show", "result", "large",
                     "also", "one", "two", "three",
                     "four", "five", "seven", "eight", "nine"]
        self.stop_words = list(stop_words.union(new_words))
        self.lmtzr = WordNetLemmatizer()

    def pre_process(self, text):
        """
        Preprocesses the input text.

        Args:
            text (str): Input text.

        Returns:
            str: Preprocessed text.
        """
        return preprocess_text(text, self.stop_words, self.lmtzr)


# Create instance of DataPreprocessor and apply preprocessing
preprocessor = DataPreprocessor()
docs = data['text'].apply(lambda x: preprocessor.pre_process(x))

class StreamlitApp:
    """
    Class for running the Streamlit App for keyword extraction.

    Attributes:
        data (pd.DataFrame): Input data containing text.
        docs (pd.Series): Preprocessed text.

    Methods:
        run(): Runs the Streamlit App.
    """
    def __init__(self, data, docs):
        """
        Initializes StreamlitApp with input data and preprocessed text.

        Args:
            data (pd.DataFrame): Input data containing text.
            docs (pd.Series): Preprocessed text.
        """
        self.data = data
        self.docs = docs

    def run(self):
        """
        Runs the Streamlit App.
        """
        st.title("Keyword Extraction App")
        idx = st.number_input("Enter an index value:", min_value=0, max_value=len(self.data)-1, value=0)
        selected_option = st.sidebar.selectbox("Select Keyword Extraction Method", ["RAKE", "YAKE", "TF-IDF"])

        keyword_extractor = KeywordExtractor(self.docs)

        if selected_option == "RAKE":
            keywords = keyword_extractor.get_keywords_rake(idx)
        elif selected_option == "YAKE":
            keywords = keyword_extractor.get_keywords_yake(idx)
        elif selected_option == "TF-IDF":
            keywords = keyword_extractor.get_keywords(idx)

        st.write("\n=====TEXT=====")
        st.write(self.data['text'][idx])
        st.write("\n===Keywords===")
        for keyword in keywords:
            st.write(keyword)

# Run the Streamlit App
streamlit_app = StreamlitApp(data, docs)
streamlit_app.run()
