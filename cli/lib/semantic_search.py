import os
import json
import numpy as np


from sentence_transformers import SentenceTransformer
from .search_utils import (
    CACHE_DIR,
    load_movies
)

class SemanticSearch():
    def __init__(self):
        self.movie_embeddings_path = os.path.join(CACHE_DIR, 'movie_embeddings.npy')
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("ValueError: text to embed is empty")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        docstr = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            docstr.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docstr, show_progress_bar=True)
        np.save(self.movie_embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        if os.path.exists(self.movie_embeddings_path):
            self.documents = documents
            for doc in self.documents:
                self.document_map[doc['id']] = doc
            self.embeddings = np.load(self.movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)


def verify_model():
    semsearch = SemanticSearch()
    print(f"Model loaded: {semsearch.model}")
    print(f"Max sequence length: {semsearch.model.max_seq_length}")
    
def verify_embeddings():
    semsearch = SemanticSearch()
    documents = load_movies()
    embeddings = semsearch.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    

def embed_text(text):
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimension: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def embed_query_text(query):
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    