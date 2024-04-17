import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora
from gensim.models import LdaModel
import umap.umap_ as umap
import plotly.express as px
import time

def preprocess_data(data):
    data = data.fillna('').astype(str)
    data['metadata'] = data['title'] + '. ' + data['summary'] + '. ' + data['overall_design'] + '. ' + data['type'] + '. ' + data['contributor']
    return data

def vectorize_data(df, method):
    if method == 'bow':
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(df['metadata'])

    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(df['metadata'])

    elif method == 'doc2vec':
        tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(df['metadata'])]
        model = Doc2Vec(tagged_docs, vector_size=100, window=5, min_count=1, workers=4)
        return np.array([model.infer_vector(doc.split()) for doc in df['metadata']])

    elif method == 'lda':
        tokenized_docs = [doc.split() for doc in df['metadata']]
        dictionary = corpora.Dictionary(tokenized_docs)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        lda_model = LdaModel(corpus, num_topics=100, id2word=dictionary, passes=20)

        topic_distributions = []
        for doc in tokenized_docs:
            topic_dist = lda_model[dictionary.doc2bow(doc)]
            topic_dist_array = np.zeros(100)
            for topic_id, prob in topic_dist:
                topic_dist_array[topic_id] = prob
            topic_distributions.append(topic_dist_array)
        return np.array(topic_distributions)    

    # sentence transformers
    else: 
        corpus = df['metadata'].tolist()
        model = SentenceTransformer(method)
        return model.encode(corpus)


def evaluate_clustering(data, embeddings, num_clusters):
    # for tf-idf and bow
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.toarray()

    km = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = km.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, clusters)
    davies_bouldin = davies_bouldin_score(embeddings, clusters)
    calinski_harabasz = calinski_harabasz_score(embeddings, clusters)

    true_labels = data['topic']
    ari = adjusted_rand_score(true_labels, clusters)
    ami = adjusted_mutual_info_score(true_labels, clusters)

    return silhouette, davies_bouldin, calinski_harabasz, ari, ami

def perform_clustering(df, method, num_clusters=8):
    embeddings = vectorize_data(df, method)
    df['cluster'] = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(embeddings)

    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(embeddings)

    df['UMAP_1'] = umap_embeddings[:, 0]
    df['UMAP_2'] = umap_embeddings[:, 1]
    df['cluster'] = df['cluster'].astype('category')

    silhouette, davies_bouldin, calinski_harabasz, ari, ami = evaluate_clustering(df, embeddings, num_clusters)
    print(f"\nResults for {method}:")
    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz}")
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Adjusted Mutual Information (AMI): {ami}")
    return df

def plot_clustering(df, method):
    fig = px.scatter(
        df,
        x='UMAP_1',
        y='UMAP_2',
        color='cluster',
        hover_data=['topic'],
        title=f'Metadata Clustering ({method})'
    )
    fig.write_html(f'plots/clustering_{method.replace("/", "_")}.html')

# Load data
df = pd.read_csv('data/processed_data.csv')
df = preprocess_data(df)

# List of methods to try
methods = [
    'all-MiniLM-L6-v2', 
    'all-MiniLM-L12-v2', 
    'all-mpnet-base-v2',
    'allenai/specter',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'tfidf',
    'bow',
    'doc2vec',
    'lda', 
    ]

for method in methods:
    start_time = time.time()
    clustered_df = perform_clustering(df, method)
    end_time = time.time()
    plot_clustering(clustered_df, method)

    print(f"Time taken: {(end_time - start_time):.2f} seconds")
    print('---')