import os
import json
import math
import pickle
import string
import itertools

from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords
)

class InvertedIndex():
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_DIR, 'index.pkl')
        self.docmap_path = os.path.join(CACHE_DIR, 'docmap.pkl')
        self.tf_path = os.path.join(CACHE_DIR, 'term_frequencies.pkl')
        self.doclengths_path = os.path.join(CACHE_DIR, 'doc_lengths.pkl')
        self.doc_lengths = {}
    
    def __get_avg_doc_length(self):
        return sum(self.doc_lengths.values()) / len(self.doc_lengths) if len(self.doc_lengths) > 0 else 0.0
    
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        if doc_id not in self.doc_lengths:
            self.doc_lengths[doc_id] = len(tokens)
        else:
            self.doc_lengths[doc_id] += len(tokens)
                
    def get_documents(self, term):
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens)>1:
            raise Exception(f"Error: expected one token: got {tokens}")
        return self.term_frequencies[doc_id][tokens[0]]
    
    def get_idf(self, term):
        token = tokenize_text(term)[0]
        doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        
        return math.log((doc_count + 1) / (term_match_doc_count + 1))
    
    def get_tf_idf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term):
        token = tokenize_text(term)[0]
        doc_count = len(self.docmap)
        df = len(self.index[token])
        return math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id]/self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf+k1 * length_norm)
    
    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
        
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = {}
        for token in tokens:
            for doc_id in self.get_documents(token):
                if doc_id not in scores:
                    scores[doc_id] = self.bm25(doc_id, token)
                else:
                    scores[doc_id] += self.bm25(doc_id, token)

        results = [(k,self.docmap[k]['title'], v) for k, v in sorted(scores.items(), key=lambda item:item[1], reverse=True)]
        return results[:limit]

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
        with open(self.tf_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doclengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)
            
    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"{self.index_path} does not exist")
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"{self.docmap_path} does not exist")
        if not os.path.exists(self.tf_path):
            raise FileNotFoundError(f"{self.tf_path} does not exist")
        if not os.path.exists(self.doclengths_path):
            raise FileNotFoundError(f"{self.doclengths_path} does not exist")
        
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doclengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def idf_command(term):
    idx = InvertedIndex()
    idx.load()    
    return idx.get_idf(term)

def tf_idf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1):
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1)

def bm25_search_command(query, limit=5):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)

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
    filtered_tokens = list(filter(lambda token: token and token not in stopwords, tokens))
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_words

