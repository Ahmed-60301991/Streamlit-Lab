import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load precomputed document embeddings [cite: 22]
# These files must be in the same folder as app.py
embeddings = np.load("embeddings.npy") [cite: 23]
with open("documents.txt", "r", encoding="utf-8") as f: [cite: 24]
    documents = f.readlines() [cite: 25]

# 2. Define the Similarity Function [cite: 26]
def retrieve_top_k(query_embedding, embeddings, k=10):
    """Retrieve top-k most similar documents using cosine similarity.""" [cite: 28]
    # Note: the [0] from the doc is used here to extract the first row of results
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0] [cite: 27, 29]
    top_k_indices = similarities.argsort()[-k:][::-1] [cite: 30]
    return [(documents[i], similarities[i]) for i in top_k_indices] [cite: 31]

# 3. Streamlit UI Components [cite: 33, 35]
st.title("Information Retrieval using Document Embeddings") [cite: 36]

# Input query from user [cite: 37]
query = st.text_input("Enter your query:") [cite: 38]

# Placeholder for actual embedding model [cite: 39, 40]
def get_query_embedding(query):
    # This generates a random vector matching your data's dimensions [cite: 41]
    return np.random.rand(embeddings.shape[1]) 

# 4. Search Logic [cite: 42]
if st.button("Search"):
    query_embedding = get_query_embedding(query) [cite: 43]
    results = retrieve_top_k(query_embedding, embeddings) [cite: 44]
    
    # Display results [cite: 45]
    st.write("### Top 10 Relevant Documents:") [cite: 46]
    for doc, score in results: [cite: 47]
        st.write(f"- **{doc.strip()}** (Score: {score:.4f})") [cite: 48]