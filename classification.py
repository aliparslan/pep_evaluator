import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
df = pd.read_csv('data/processed_data.csv')
df = df.fillna('').astype(str)

# Combine various columns into one 'metadata' column
df['metadata'] = df['title'] + '. ' + df['summary'] + '. ' + df['overall_design'] + '. ' + df['type'] + '. ' + df['contributor']

# Preprocess the text data
df['metadata'] = df['metadata'].str.lower()
df['metadata'] = df['metadata'].str.replace('[^a-zA-Z0-9\s]', '')

# Split the comma-separated topics into lists
df['mesh_terms'] = df['mesh_terms'].apply(lambda x: x.split(','))

# Create a MultiLabelBinarizer object
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(df['mesh_terms'])

# Extract features using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(df['metadata'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a OneVsRestClassifier with Logistic Regression as the base estimator
classifier = OneVsRestClassifier(LogisticRegression())
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
f1 = f1_score(y_test, y_pred, average='micro')
print(f"F1 Score: {f1:.4f}")

# Predict topics for new text
new_text = ["Lining macrophages initiate articular inflammation. This dataset contains the transcriptional analysis of lining macrophages and sub-lining macrophages from vehicle or antigen-induced arthritis(AIA) mouse model. mice were immunized by two subcutaneous injection of 20 µg mBSA (40 mg/ml) emulsified with CFA (3.3 mg/ml) and PBS to a final 100 µl volume per mouse. Two hundred lining and sublining macrophages from 3 WT and 3 IRF5KO mice, isolated from PBS-Mbsa injected knees. Sorting of synovial macrophages was carried out from the knee cell suspensions. Lining macrophages were defined as CD45+, Linage- (NK1.1, CD3, CD19), CD11b+, F4/80+ and VSIG4+ cells, whereas sublining macrophages were CD45+,  Lin-, CD11b+, F4/80+ and VSIG4-."]
new_features = vectorizer.transform(new_text)
predicted_labels = classifier.predict(new_features)

# Map the predicted labels back to topic terms
predicted_topics = mlb.inverse_transform(predicted_labels)
print(f"Predicted Topics: {predicted_topics}")