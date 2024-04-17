import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.preprocessing import MultiLabelBinarizer
from annoy import AnnoyIndex

# Load the dataset
df = pd.read_csv('data/processed_data.csv')
df = df.fillna('').astype(str)

# Combine various columns into one 'metadata' column
df['metadata'] = df['title'] + '. ' + df['summary'] + '. ' + df['overall_design'] + '. ' + df['type'] + '. ' + df['contributor']

# Load the pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode the metadata using BERT
embeddings = model.encode(df['metadata'].tolist())

# Normalize the embeddings
embeddings = normalize(embeddings)

# Initialize Annoy index
vector_dim = embeddings.shape[1]
annoy_index = AnnoyIndex(vector_dim, metric='angular')

# Add dataset embeddings to the Annoy index
for i, embedding in enumerate(embeddings):
    annoy_index.add_item(i, embedding)

# Build the Annoy index
num_trees = 10
annoy_index.build(num_trees)

# Define the queries and relevant dataset IDs
queries = [
    "scRNA in bone",
    "blood in mice",
    "cancer biomarkers",
    "immune response to infection",
    "neurodegenerative disorders"
]
relevant_gses = [
    ["GSE123456", "GSE789012", "GSE345678"],
    ["GSE901234", "GSE567890", "GSE234567"],
    ["GSE890123", "GSE456789", "GSE012345"],
    ["GSE678901", "GSE234567", "GSE890123"],
    ["GSE567890", "GSE123456", "GSE901234"]
]

# Encode the queries using the same embedding model
query_embeddings = model.encode(queries)

# Perform retrieval for each query
for query, query_embedding, relevant_gse in zip(queries, query_embeddings, relevant_gses):
    # Retrieve the top-k most similar datasets
    k = 10
    retrieved_indices = annoy_index.get_nns_by_vector(query_embedding, k)
    
    # Get the retrieved dataset IDs and their corresponding MeSH terms
    retrieved_gses = df.iloc[retrieved_indices]['gse'].tolist()
    retrieved_mesh_terms = df.iloc[retrieved_indices]['mesh_terms'].tolist()
    
    # Pad the retrieved dataset IDs if needed
    if len(retrieved_gses) < len(relevant_gse):
        retrieved_gses += [''] * (len(relevant_gse) - len(retrieved_gses))
    
    # Convert the labels into multilabel format
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform([[gse] for gse in relevant_gse])
    y_pred = mlb.transform([[gse] for gse in retrieved_gses])

    # Compute evaluation metrics
    average_precision = average_precision_score(y_true, y_pred)
    ndcg = ndcg_score(y_true, y_pred)

    print(f"Query: {query}")
    print(f"Retrieved datasets: {retrieved_gses}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"NDCG: {ndcg:.4f}")
    print()