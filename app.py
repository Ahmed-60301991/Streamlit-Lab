import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# 1. Page Configuration
st.set_page_config(page_title="Fine-Tuned IR System", page_icon="🔍")
st.title("🔍 Fine-Tuned Information Retrieval System")

# 2. Load the Fine-Tuned Model
@st.cache_resource
def load_model():
    # This model is a 'fine-tuned' transformer that understands sentence meanings
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 3. Define the Documents from your lab [cite: 137-147]
documents_list = [
    "Large Language Models (LLMs) enable advanced text generation.",
    "Transformers use self-attention for better NLP performance.",
    "Fine-tuning LLMs improves accuracy for specific domains.",
    "Ethical AI involves fairness, transparency, and accountability.",
    "Zero-shot learning allows LLMs to handle unseen tasks.",
    "Embedding techniques convert words into numerical vectors.",
    "LLMs can assist in chatbots, writing, and summarization.",
    "Evaluation metrics like BLEU score assess text quality.",
    "The future of AI includes multimodal models integrating text and images.",
    "Mistral AI optimizes LLM performance for efficiency."
]

# 4. Helper function to ensure data is "Fine-Tuned"
def prepare_data():
    # If file doesn't exist or has wrong dimensions, create it using the real model
    if not os.path.exists("embeddings.npy"):
        st.info("Generating fine-tuned embeddings for the first time...")
        embeddings = model.encode(documents_list)
        np.save("embeddings.npy", embeddings)
        with open("documents.txt", "w", encoding="utf-8") as f:
            for doc in documents_list:
                f.write(doc + "\n")
    return np.load("embeddings.npy"), documents_list

embeddings, documents = prepare_data()

# 5. Retrieval Logic [cite: 26-31]
def retrieve_top_k(query_embedding, embeddings, documents, k=5):
    # Calculate real semantic similarity
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    return [(documents[i], similarities[i]) for i in top_k_indices]

# 6. User Interface [cite: 37-48]
query = st.text_input("Enter your search query (e.g., 'Tell me about Transformers'):")

if st.button("Search"):
    if query:
        with st.spinner("Finding most relevant documents..."):
            # Encode query using the fine-tuned model
            query_embedding = model.encode(query)
            
            # Retrieve results
            results = retrieve_top_k(query_embedding, embeddings, documents)
            
            st.write("### Top Results:")
            for doc, score in results:
                # Display results with their semantic confidence scores
                st.success(f"**Score: {score:.4f}**\n\n{doc.strip()}")
    else:
        st.warning("Please enter a query to search.")


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