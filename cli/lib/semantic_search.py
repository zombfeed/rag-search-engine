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
        
    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loadead. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in similarities[:limit]:
            results.append({
                'title': doc['title'],
                'description': doc['description'],
                'score': score
            })
        return results

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
    
def semantic_search(query, limit):
    semsearch = SemanticSearch()
    documents = load_movies()
    semsearch.load_or_create_embeddings(documents)
    results = semsearch.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...\n")

def cosine_similarity(vecA, vecB):
    dot_product = np.dot(vecA, vecB)
    normA = np.linalg.norm(vecA)
    normB = np.linalg.norm(vecB)
    if normA == 0 or normB == 0:
        return 0.0
    return dot_product / (normA * normB)