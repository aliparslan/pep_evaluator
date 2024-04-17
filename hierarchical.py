import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import plotly.express as px

def preprocess_data(data):
    data = data.fillna('').astype(str)
    data['metadata'] = data['title'] + '. ' + data['summary'] + '. ' + data['overall_design'] + '. ' + data['type'] + '. ' + data['contributor']
    data['metadata'] = data['metadata'].str.lower()
    return data

def evaluate_clustering(data, embeddings, num_clusters):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    clusters = clustering.fit_predict(embeddings)
    
    silhouette = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)
    calinski_harabasz = calinski_harabasz_score(embeddings, clusters)
    
    true_labels = data['topic']
    ari = adjusted_rand_score(true_labels, clusters)
    ami = adjusted_mutual_info_score(true_labels, clusters)
    
    return silhouette, davies_bouldin, calinski_harabasz, ari, ami

# Load data
df = pd.read_csv('data/processed_data.csv')
df = preprocess_data(df)

# generate embeddings
model = SentenceTransformer('miniLM-L6-v2')
embeddings = model.encode(df['metadata'].tolist())

# Hierarchical clustering
num_clusters = 8
clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
df['cluster'] = clustering.fit_predict(embeddings)

# UMAP dimension reduction
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = umap_reducer.fit_transform(embeddings)

silhouette, davies_bouldin, calinski_harabasz, ari, ami = evaluate_clustering(df, embeddings, num_clusters)
print(f"Silhouette Score: {silhouette}")
print(f"Davies-Bouldin Index: {davies_bouldin}")
print(f"Calinski-Harabasz Index: {calinski_harabasz}")
print(f"Adjusted Rand Index (ARI): {ari}")
print(f"Adjusted Mutual Information (AMI): {ami}")

# Adding UMAP dimensions to DataFrame
df['UMAP_1'] = umap_embeddings[:, 0]
df['UMAP_2'] = umap_embeddings[:, 1]

# Convert 'cluster' column to categorical for better colors
df['cluster'] = df['cluster'].astype('category')

# Create a plot
fig = px.scatter(
    df,
    x='UMAP_1',
    y='UMAP_2',
    color='cluster',
    hover_data=['topic'],
    title='Metadata Clustering (Hierarchical)'
)

fig.show()