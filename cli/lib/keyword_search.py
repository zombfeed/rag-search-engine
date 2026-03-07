import os
import json
import pickle
import string
import argparse

from collections import defaultdict
from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords
)

class InvertedIndex():
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_DIR, 'docmap.pkl')
    
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
                
    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def build(self):
        movies = load_movies()
  
        for movie in movies:
            doc_id = movie['id']
            doc_desc = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_desc)
    
    def save(self):
        os.makedirs('cache', exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
            
    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"{self.index_path} does not exist")
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"{self.docmap_path} does not exist")
        
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    seen, results = set(), []
    try:
        idx.load()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return results

    query_tokens = tokenize_text(query)
    for token in query_tokens:
        matching_doc_ids = idx.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(idx.docmap[doc_id])
            if len(results) >= limit:
                break
            
    return results
        
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
    
def tokenize_text(text):
    text = preprocess_text(text)
    tokens = text.split()
    stopwords = load_stopwords()
    filtered_tokens = list(filter(lambda token: token or token not in stopwords, tokens))
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_words
    
    

def formatresults(results):
    results.sort(key=lambda x: x['id'])
    for i in range(len(results)):
        print(f"{i+1}. {results[i]['title']}")

def loadFromJson(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
