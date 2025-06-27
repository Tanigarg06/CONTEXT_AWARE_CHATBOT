import streamlit as st
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import textwrap
import fitz  # PyMuPDF for PDF handling
import google.generativeai as genai


#Google gemini API KEY
API_KEY = st.secrets["GEMINI"]["API_KEY"]
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")


# streamlit configuration
st.set_page_config(page_title="GEMINI CHATBOT",layout="centered")
st.title("Retrieval-Based Chatbot with Gemini")

if "history" not in st.session_state:
    st.session_state.history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.chunks_vectors = None

def chunk_text(text, chunk_size=300):
    """Split text into chunks of a specified size."""
    return textwrap.wrap(text, width=chunk_size)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_relevant_chunks(query):
    """Retrieve relevant chunks based on the query."""
    vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(vec, st.session_state.chunks_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return "\n\n".join([st.session_state.chunks[i] for i in top_indices])
def build_prompt(query):
    chat_history = "\n".join(
        [f"User: {q}\nBot: {a}" for q, a in st.session_state.history[-3:]]
    )
    context = get_relevant_chunks(query)
    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Chat History:
{chat_history}

User: {query}
Bot:"""
    return prompt


def process_file(uploaded_file, file_type):
    """Process the uploaded file and extract text."""
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")
    # Chunk the text
    chunks = chunk_text(text)
    vectorizer= TfidfVectorizer().fit(chunks)
    chunks_vectors = vectorizer.transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks_vectors = chunks_vectors

st.sidebar.header("Upload file")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded_file:
    file_type = uploaded_file.type.split("/")[-1]
    process_file(uploaded_file, file_type)
    st.sidebar.success("File uploaded successfully!")    

# CHAT INTERFACE
st.sidebar.header("ASK QUERIES")    
user_query = st.text_input("Your question:")

if st.button("ASK") and user_query and st.session_state.vectorizer:
    # Build the prompt
    prompt = build_prompt(user_query)
    try:
        response = model.generate_content(prompt)
        bot_reply = response.text.strip()
        st.session_state.history.append((user_query, bot_reply))
        st.success("Response generated successfully!")
    except Exception as e:
        st.error(f"Error generating response: {e}")

if st.session_state.history:
    st.subheader("Conversation")
    for user, bot in reversed(st.session_state.history):
        st.markdown(f"**User:** {user}")
        st.markdown(f"**Bot:** {bot}")
        st.markdown("---")
        
