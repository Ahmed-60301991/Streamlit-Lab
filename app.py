import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set page config
st.set_page_config(page_title="Fine-Tuned IR App", layout="centered")

# 1. Load the fine-tuned embedding model
# We use st.cache_resource so the model only loads once, saving memory
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 2. Load the document embeddings and text [cite: 22-25]
embeddings = np.load("embeddings.npy")
with open("documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# 3. Retrieval Logic [cite: 26-31]
def retrieve_top_k(query_embedding, embeddings, k=5):
    """Retrieve top-k most similar documents using cosine similarity."""
    # Compute similarity between query and all stored embeddings
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    # Get indices of the highest scores
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# 4. Streamlit UI [cite: 33-38]
st.title("🚀 Fine-Tuned Information Retrieval")
st.markdown("This app uses **Sentence-Transformers** to find semantically relevant documents.")

query = st.text_input("Enter your search query:")

# 5. Execution [cite: 42-48]
if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            # Use the AI model to encode the user's query
            query_embedding = model.encode(query)
            # Retrieve matches
            results = retrieve_top_k(query_embedding, embeddings)
            
            st.write("### Top Results:")
            for doc, score in results:
                # Display the document and its confidence score
                st.info(f"{doc.strip()} \n\n **Match Score:** {score:.4f}")
    else:
        st.warning("Please enter a query to begin.")


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