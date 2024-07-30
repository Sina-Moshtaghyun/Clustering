import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from collections import Counter

# Load the Excel file
file_path = 'questions_categories.xlsx'
df = pd.read_excel(file_path)

# Assuming the columns are 'Question' and 'Category'
questions = df['question'].astype(str).tolist()
categories = df['categories'].tolist()

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = AutoModel.from_pretrained('bert-base-multilingual-uncased')


# Function to get embeddings for a list of sentences
def get_embeddings(sentences, batch_size=32):
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token representation
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)


# Convert questions to embeddings
embeddings = get_embeddings(questions)
# Number of clusters
num_clusters = len(set(categories))

# Initialize the k-means clustering
kmeans = faiss.Kmeans(d=embeddings.shape[1], k=num_clusters, niter=20, verbose=True)

# Train the clustering model
kmeans.train(embeddings)

# Get cluster assignments for each question
_, cluster_assignments = kmeans.index.search(embeddings, 1)
cluster_assignments = cluster_assignments.flatten()

# Create a mapping from cluster numbers to category names
cluster_to_questions = {i: [] for i in range(num_clusters)}
for i, cluster in enumerate(cluster_assignments):
    cluster_to_questions[cluster].append(questions[i])


# Function to extract keywords or phrases
def extract_keywords_tfidf(questions, num_keywords=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(questions)
    indices = np.argsort(vectorizer.idf_)[::-1]
    features = vectorizer.get_feature_names_out()
    top_keywords = [features[i] for i in indices[:num_keywords]]
    return top_keywords


# Auto-generate cluster names
cluster_to_category = {}
for cluster, qs in cluster_to_questions.items():
    keywords = extract_keywords_tfidf(qs)
    cluster_name = ', '.join(keywords)
    cluster_to_category[cluster] = cluster_name

# Assign cluster names to the dataframe
df['Cluster'] = [cluster_to_category[cluster] for cluster in cluster_assignments]

# Save the clustered data
df.to_excel('clustered_questions.xlsx', index=False)


def categorize_new_questions(new_questions):
    # Convert new questions to embeddings
    new_embeddings = get_embeddings(new_questions)

    # Assign new questions to the nearest cluster
    _, new_cluster_assignments = kmeans.index.search(new_embeddings, 1)
    new_cluster_assignments = new_cluster_assignments.flatten()

    # Map cluster numbers to category names
    new_cluster_names = [cluster_to_category[cluster] for cluster in new_cluster_assignments]

    # Count occurrences of each category
    cluster_counts = Counter(new_cluster_names)

    return cluster_counts