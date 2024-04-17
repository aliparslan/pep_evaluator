import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(data):
    data = data.fillna('').astype(str)
    data['metadata'] = data['title'] + '. ' + data['summary'] + '. ' + data['overall_design'] + '. ' + data['type'] + '. ' + data['contributor']
    data['metadata'] = data['metadata'].str.lower()
    return data

df = pd.read_csv('data/processed_data.csv')
df = preprocess_data(df)

models = [
    'pritamdeka/S-PubMedBERT-MS-MARCO',
    'allenai/specter'
]
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('tavakolih/all-MiniLM-L6-v2-pubmed-full')

documents = df['metadata']
document_vectors = model.encode(documents)

query = "dna methylation in adrenocortical carinoma"
query_vector = model.encode([query])[0]

similarities = cosine_similarity([query_vector], document_vectors)[0]

top_k = 3
top_doc_indices = similarities.argsort()[-top_k:][::-1]

for doc_index in top_doc_indices:
    print('Title: ', df['title'].iloc[doc_index])
    print('GSE: ', df['gse'].iloc[doc_index])
    print('Similarity: ', similarities[doc_index])
    print()