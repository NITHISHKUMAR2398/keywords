import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from rake_nltk import Rake
import yake

def download_nltk_data():
    """
    Downloads NLTK data for wordnet and stopwords.
    """
    nltk.download('wordnet')
    nltk.download('stopwords')
  
def preprocess_text(text, stop_words, lemmatizer):
    """
    Preprocesses the input text.

    Args:
        text (str): The input text to be preprocessed.
        stop_words (list): List of stopwords.
        lemmatizer (WordNetLemmatizer): NLTK WordNet lemmatizer object.

    Returns:
        str: Preprocessed text.
    """
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = [word for word in text if len(word) >= 3]
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

class KeywordExtractor:
    """
    Class for extracting keywords from a collection of documents.
    """
    def __init__(self, docs):
        """
        Initializes the KeywordExtractor.

        Args:
            docs (list): List of documents for keyword extraction.
        """
        self.docs = docs

        cv = CountVectorizer(max_df=0.95, max_features=10000, ngram_range=(1,3))
        word_count_vector = cv.fit_transform(self.docs)

        tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        self.feature_names = cv.get_feature_names_out()

    def sort_coo(self, coo_matrix):
        """
        Sorts a COO matrix.

        Args:
            coo_matrix (scipy.sparse.coo_matrix): COO matrix.

        Returns:
            list: Sorted tuples.
        """
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(self, sorted_items, topn=10):
        """
        Extracts top N items from a sorted vector.

        Args:
            sorted_items (list): Sorted items.
            topn (int, optional): Number of top items to extract. Defaults to 10.

        Returns:
            dict: Dictionary of top items.
        """
        sorted_items = sorted_items[:topn]
        score_vals = []
        feature_vals = []

        for idx, score in sorted_items:
            fname = self.feature_names[idx]
            score_vals.append(round(score, 3))
            feature_vals.append(self.feature_names[idx])

        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

    def get_keywords(self, idx):
        """
        Gets keywords for a specific document.

        Args:
            idx (int): Index of the document.

        Returns:
            dict: Dictionary of keywords and their scores.
        """
        cv = CountVectorizer(max_df=0.95, max_features=10000, ngram_range=(1,3))
        word_count_vector = cv.fit_transform(self.docs)

        tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        tf_idf_vector=tfidf_transformer.transform(cv.transform([self.docs[idx]]))
        sorted_items=self.sort_coo(tf_idf_vector.tocoo())
        keywords=self.extract_topn_from_vector(sorted_items,10)
        return keywords

    def get_keywords_yake(self, idx):
        """
        Gets keywords using YAKE algorithm for a specific document.

        Args:
            idx (int): Index of the document.

        Returns:
            list: List of tuples containing keywords and their scores.
        """
        y = yake.KeywordExtractor(lan='en', n=1, dedupLim=0.9, dedupFunc='seqm', windowsSize=1, top=10, features=None)
        keywords = y.extract_keywords(self.docs[idx])
        return keywords

    def get_keywords_rake(self, idx, n=10):
        """
        Gets keywords using RAKE algorithm for a specific document.

        Args:
            idx (int): Index of the document.
            n (int, optional): Number of top keywords to extract. Defaults to 10.

        Returns:
            list: List of top keywords.
        """
        r = Rake()
        r.extract_keywords_from_text(self.docs[idx])
        keywords = r.get_ranked_phrases()[0:n]
        return keywords
