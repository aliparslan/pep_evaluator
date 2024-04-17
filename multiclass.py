import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import jaccard_score, hamming_loss
import warnings
import time
import os

# ignore warnings saying labels are not present in all examples
warnings.filterwarnings('ignore')

# grab working directory
current_dir = os.path.join(os.getcwd(), 'data')

def preprocess_data(data):
    data = data.fillna('').astype(str)
    data = data.apply(lambda x: x.str.lower())
    data['metadata'] = data['title'] + '. ' + data['summary'] + '. ' + data['overall_design'] + '. ' + data['type'] + '. ' + data['contributor']

    # preprocess mesh_terms column
    data['mesh_terms'] = data['mesh_terms'].apply(lambda x: x.split(', '))
    return data

def vectorize_data(df, method):
    corpus = df['metadata'].tolist()
    if method == 'bow':
        vectorizer = CountVectorizer()
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid method. Please choose between 'bow' and 'tfidf'.")
    X = vectorizer.fit_transform(corpus)
    return X

def train_and_evaluate_model(model_name, df, test_size=0.2, random_state=42):
    start_time = time.time()

    # generate embeddings using sentence transformer model
    model = SentenceTransformer(model_name)
    print(f"Generating embeddings using {model_name}...")
    embeddings = model.encode(df['metadata'].tolist())

    # binarize the mesh_terms using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    # mlb = LabelEncoder()
    labels = mlb.fit_transform(df['mesh_terms'])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)

    # create and train the multi-label classification model
    print(f"Training the model using {model_name}...")
    # classifier = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, random_state=42)
    classifier = OneVsRestClassifier(LogisticRegression(random_state=random_state))
    classifier.fit(X_train, y_train)

    # predict labels for test set
    y_pred = classifier.predict(X_test)

    # evaluate results
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    jaccard = jaccard_score(y_test, y_pred, average='weighted')
    hamming = hamming_loss(y_test, y_pred)

    end_time = time.time()

    print(f"\nResults for {model_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Jaccard Index: {jaccard:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print('---')

    # predicting on labels
    df['predicted_mesh_terms'] = mlb.inverse_transform(classifier.predict(embeddings))

    # writing test data
    output_df = df[['topic', 'gse', 'pmid', 'metadata', 'mesh_terms', 'predicted_mesh_terms']]
    # output_df.to_csv(os.path.join(current_dir, f"{model_name}.csv", index=False))
    output_df.to_csv(f'outputs/test_data_{model_name.replace("/", "_")}.csv')

def train_and_evaluate_vectorizer(vectorization_method, df, test_size=0.2, random_state=42):
    start_time = time.time()

    # generate embeddings using sentence transformer model
    X = vectorize_data(df, vectorization_method)

    # binarize the mesh_terms using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    # mlb = LabelEncoder()
    labels = mlb.fit_transform(df['mesh_terms'])

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state)

    # create and train the multi-label classification model
    print(f"Training the model using {vectorization_method}...")
    # classifier = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000, random_state=42)
    classifier = OneVsRestClassifier(LogisticRegression(random_state=random_state))
    classifier.fit(X_train, y_train)

    # predict labels for test set
    y_pred = classifier.predict(X_test)

    # evaluate results
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    jaccard = jaccard_score(y_test, y_pred, average='weighted')
    hamming = hamming_loss(y_test, y_pred)

    end_time = time.time()

    print(f"\nResults for {vectorization_method}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Jaccard Index: {jaccard:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print('---')

    # predicting on labels
    df['predicted_mesh_terms'] = mlb.inverse_transform(classifier.predict(X))

    # # writing test data
    # output_df = df[['topic', 'gse', 'pmid', 'metadata', 'mesh_terms', 'predicted_mesh_terms']]
    # # output_df.to_csv(os.path.join(current_dir, f"{model_name}.csv", index=False))
    # output_df.to_csv(f'outputs/test_data_{vectorization_method}.csv')

# Load data
df = pd.read_csv('data/processed_data.csv')
df = preprocess_data(df)

embedding_models = [
    'all-MiniLM-L6-v2',
    # 'all-MiniLM-L12-v2',
    # 'all-mpnet-base-v2',
    # 'distilbert/distilbert-base-uncased',
    # 'distilbert/distilroberta-base',
    # 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    # 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    # 'allenai/biomed_roberta_base',
    # 'allenai-specter',
    'allenai/specter2_base',
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
]

for model in embedding_models:
    train_and_evaluate_model(model, df)

# train_and_evaluate_vectorizer('bow', df)
# train_and_evaluate_vectorizer('tfidf', df)