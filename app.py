import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = get_model()

# Load files
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r") as f:
    documents = f.readlines()

st.title("🔍 Real Semantic Search")
query = st.text_input("Search for something (e.g., 'machine learning'):")

if query:
    # No more random! We encode the real query.
    query_embedding = model.encode([query]) 
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_idx = similarities.argsort()[-5:][::-1]

    for idx in top_k_idx:
        st.write(f"Score: {similarities[idx]:.4f}")
        st.info(documents[idx])


# import streamlit as st
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # Load precomputed document embeddings
# embeddings = np.load("embeddings.npy")
# with open("documents.txt", "r", encoding="utf-8") as f:
#     documents = f.readlines()

# def retrieve_top_k(query_embedding, embeddings, k=10):
#     """Retrieve top-k most similar documents using cosine similarity."""
#     similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
#     top_k_indices = similarities.argsort()[-k:][::-1]
#     return [(documents[i], similarities[i]) for i in top_k_indices]

# # Streamlit UI
# st.title("Information Retrieval using Document Embeddings")

# # Input query
# query = st.text_input("Enter your query:")

# # Load or compute query embedding
# def get_query_embedding(query):
#     return np.random.rand(embeddings.shape[1]) 

# if st.button("Search"):
#     query_embedding = get_query_embedding(query)
#     results = retrieve_top_k(query_embedding, embeddings)
    
#     # Display results
#     st.write("### Top 10 Relevant Documents:")
#     for doc, score in results:
#         st.write(f"- **{doc.strip()}** (Score: {score:.4f})")