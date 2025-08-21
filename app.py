import warnings
warnings.filterwarnings("ignore", message=".*preloaded with link preload was not used.*")
warnings.filterwarnings("ignore")

import streamlit as st
import os
import sys
import io
import json
import hashlib
import requests
import tempfile
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import fitz
    from docx import Document
    from email import policy
    from email.parser import BytesParser
except ImportError as e:
    st.error(f"Document processing library missing: {e}")
    st.stop()

try:
    import google.generativeai as genai
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
except ImportError as e:
    st.error(f"AI/ML library missing: {e}")
    st.stop()

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    with st.spinner("Downloading language data..."):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

GENAI_API_KEY = "AIzaSyBr6aFYhZSuUJbWvi07NGHvtK1q3JnUWIg"
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 5

genai.configure(api_key=GENAI_API_KEY)

st.set_page_config(
    page_title="📚 Intelligent Q&A System by Intelligentsia",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --background-color: #f5f6fa;
            --surface-color: #ffffff;
            --text-color: #2c3e50;
            --secondary-text: #7f8c8d;
            --accent-color: #e74c3c;
            --shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .block-container {
            padding: 1.5rem 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        .main {
            background-color: var(--background-color);
            font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-color);
        }

        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 2rem 3rem;
            border-radius: 0 0 15px 15px;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
            position: sticky;
            top: 0;
            z-index: 1000;
            animation: fadeInDown 0.6s ease-out;
        }

        header > div {
            display: flex;
            align-items: center;
            justify-content: space-between;
            max-width: 1400px;
            margin: 0 auto;
        }

        header h1 {
            margin: 0;
            font-size: 2.5rem;
            color: #fff;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        header .header-icon {
            font-size: 1.5rem;
            color: #fff;
        }

        .sidebar {
            background-color: var(--surface-color);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: var(--shadow);
            height: 100%; /* Remove calc and fixed height to eliminate empty space */
            position: sticky;
            top: 2rem;
            animation: slideInLeft 0.5s ease-out;
        }

        .sidebar .sidebar-section {
            margin-bottom: 2rem;
        }

        .sidebar .sidebar-section h3 {
            font-size: 1.2rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
            font-weight: 600;
        }

        .stFileUploader, .stTextInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #ddd;
            padding: 0.9rem;
            background-color: #f8f9fa;
            color: var(--text-color);
            font-size: 1rem;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        .stFileUploader:hover, .stTextInput > div > div > input:hover {
            border-color: var(--secondary-color);
            box-shadow: 0 0 8px rgba(52, 152, 219, 0.3);
        }

        div.stButton > button {
            background: var(--secondary-color);
            color: #fff;
            border-radius: 10px;
            padding: 0.9rem 1.8rem;
            font-weight: 500;
            font-size: 1rem;
            border: none;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
        }

        div.stButton > button:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        }

        input[type="text"], textarea {
            border-radius: 10px;
            border: 1px solid #ddd;
            padding: 0.9rem;
            background-color: #f8f9fa;
            color: var(--text-color);
            font-size: 1rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }

        textarea[disabled] {
            background-color: #ecf0f1 !important;
            color: var(--secondary-text);
        }

        .stAlert {
            border-radius: 10px;
            background-color: #fff;
            color: var(--text-color);
            padding: 1.2rem;
            font-size: 0.95rem;
            border-left: 4px solid var(--accent-color);
        }

        .content {
            background: var(--surface-color);
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: var(--shadow);
            animation: fadeIn 0.8s ease-out;
        }

        .content p {
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }

        h1, h2, h3, h4 {
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
            color: var(--primary-color);
        }

        .subheading {
            font-size: 1.3rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem;
            color: var(--primary-color);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 0.5rem;
            display: inline-block;
        }

        .chat-container {
            max-height: 550px;
            overflow-y: auto;
            padding: 1.5rem;
            border-radius: 15px;
            background-color: #f8f9fa;
            margin-bottom: 1.5rem;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.1);
        }

        .chat-message {
            display: flex;
            margin-bottom: 1.5rem;
            animation: slideInUp 0.4s ease-out;
        }

        .chat-message.user {
            justify-content: flex-end;
        }

        .chat-message.bot {
            justify-content: flex-start;
        }

        .chat-bubble {
            max-width: 70%;
            padding: 1.2rem;
            border-radius: 15px;
            font-size: 1rem;
            line-height: 1.6;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        .chat-message.user .chat-bubble {
            background: var(--secondary-color);
            color: #fff;
            border-bottom-right-radius: 4px;
        }

        .chat-message.bot .chat-bubble {
            background: #ecf0f1;
            color: var(--text-color);
            border-bottom-left-radius: 4px;
        }

        div.stExpander {
            background-color: #f8f9fa;
            border-radius: 15px;
            box-shadow: var(--shadow);
            margin-top: 1.5rem;
        }

        div.stExpander[expanded="true"] > div {
            border-top: 2px solid var(--secondary-color);
        }

        .stTextInput > div > div {
            margin-bottom: 0;
        }

        footer {
            text-align: center;
            padding: 1.5rem 0;
            background-color: var(--primary-color);
            color: #fff;
            font-size: 0.9rem;
            margin-top: 2rem;
            border-radius: 15px 15px 0 0;
            box-shadow: var(--shadow);
        }

        footer a {
            color: var(--secondary-color);
            text-decoration: none;
            font-weight: 600;
        }

        footer a:hover {
            text-decoration: underline;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }

        @keyframes slideInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    defaults = {
        'embeddings': None,
        'chunks': [],
        'vectorizer': None,
        'tfidf_matrix': None,
        'source_type': None,
        'source_name': '',
        'is_loaded': False,
        'question': '',
        'response': None,
        'chat_history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    if not text.strip():
        return []
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def preprocess_text(text: str) -> str:
    return ' '.join(text.split())

def extract_pdf_text(file_content: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        text = ""
        with fitz.open(tmp_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        os.unlink(tmp_path)
        return preprocess_text(text)
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return ""

def extract_docx_text(file_content: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        doc = Document(tmp_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        os.unlink(tmp_path)
        return preprocess_text(text)
    except Exception as e:
        st.error(f"Error extracting DOCX: {e}")
        return ""

def extract_eml_text(file_content: bytes) -> str:
    try:
        msg = BytesParser(policy=policy.default).parsebytes(file_content)
        text = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    text += part.get_content()
        else:
            text = msg.get_content()
        return preprocess_text(text)
    except Exception as e:
        st.error(f"Error extracting EML: {e}")
        return ""

def extract_website_text(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return preprocess_text(text)
    except Exception as e:
        st.error(f"Error extracting website: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

def create_embeddings(chunks: List[str]) -> Optional[np.ndarray]:
    if not chunks:
        return None
    try:
        model = load_embedding_model()
        if model is None:
            return None
        embeddings = model.encode(chunks, show_progress_bar=False)
        return embeddings
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

@st.cache_resource(show_spinner=False)
def create_tfidf_matrix_cached(chunks: List[str]):
    try:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True)
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"Error creating TF-IDF matrix: {e}")
        return None, None

def semantic_search(query: str, embeddings: np.ndarray, chunks: List[str], top_k: int = TOP_K) -> List[str]:
    try:
        model = load_embedding_model()
        if model is None or embeddings is None:
            return chunks[:top_k]
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        st.error(f"Error in semantic search: {e}")
        return chunks[:top_k]

def keyword_search(query: str, vectorizer, tfidf_matrix, chunks: List[str], top_k: int = TOP_K) -> List[str]:
    try:
        if vectorizer is None or tfidf_matrix is None:
            return chunks[:top_k]
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        st.error(f"Error in keyword search: {e}")
        return chunks[:top_k]

def hybrid_search(query: str) -> List[str]:
    if not st.session_state.chunks:
        return []
    semantic_results = semantic_search(query, st.session_state.embeddings, st.session_state.chunks)
    keyword_results = keyword_search(query, st.session_state.vectorizer, st.session_state.tfidf_matrix, st.session_state.chunks)
    combined = []
    seen = set()
    for result in semantic_results + keyword_results:
        if result not in seen:
            combined.append(result)
            seen.add(result)
    return combined[:TOP_K]

def get_ai_response_with_memory(query: str, user_name: str = "User") -> Dict:
    try:
        memory_context = "\n".join([f"{item['user']}: {item['message']}\nBot: {item['response']}" 
                                    for item in st.session_state.chat_history])
        relevant_chunks = hybrid_search(query)
        context_text = ""
        if relevant_chunks:
            context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(relevant_chunks)])
        
        prompt = f"""
Hi {user_name}! You asked:

{query}

Here is your chat history:
{memory_context}

Based on the document context below, please answer in a friendly and interactive way:
{context_text}

Please provide a detailed answer and cite sources when possible.
Answer:"""
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        answer_text = response.text.strip()
        
        st.session_state.chat_history.append({
            "user": user_name,
            "message": query,
            "response": answer_text
        })
        
        confidence = "high" if len(answer_text) > 100 and len(relevant_chunks) >= 3 else "medium" if len(answer_text) > 50 else "low"
        
        return {
            "answer": answer_text,
            "confidence": confidence,
            "sources_used": len(relevant_chunks),
            "context": relevant_chunks[:2]
        }
        
    except Exception as e:
        return {
            "answer": f"Hi {user_name}, I ran into an error while generating the response: {str(e)}",
            "confidence": "low",
            "sources_used": 0,
            "context": []
        }

def process_source(source_type: str, content, name: str = ""):
    try:
        if source_type == "pdf":
            text = extract_pdf_text(content)
        elif source_type == "docx":
            text = extract_docx_text(content)
        elif source_type == "eml":
            text = extract_eml_text(content)
        elif source_type == "website":
            text = extract_website_text(content)
        else:
            return False, "Unsupported source type"
        if not text.strip():
            return False, "No text content found"
        chunks = chunk_text(text)
        if not chunks:
            return False, "Could not create text chunks"
        with st.spinner("Creating embeddings..."):
            embeddings = create_embeddings(chunks)
        with st.spinner("Creating search index..."):
            vectorizer, tfidf_matrix = create_tfidf_matrix_cached(chunks)
        st.session_state.embeddings = embeddings
        st.session_state.chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.source_type = source_type
        st.session_state.source_name = name
        st.session_state.is_loaded = True
        return True, f"Successfully processed {len(chunks)} text chunks"
    except Exception as e:
        return False, f"Error processing source: {str(e)}"

def main():
    st.markdown("""
        <header>
            <div>
                <h1>📚 Intelligent Q&A System by Intelligentsia</h1>
                <div class="header-icon">🤖</div>
            </div>
        </header>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="content">
            <p style="font-size:1.1rem; color:var(--secondary-text); line-height:1.6;">
                Welcome to our advanced AI-powered Q&A system. Upload documents or provide website URLs to extract information, then engage with the chatbot to get accurate answers with context-aware responses.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3>📊 Dashboard</h3>', unsafe_allow_html=True)
        if st.session_state.is_loaded:
            st.markdown(f'<p style="color:var(--secondary-text);">Processed Chunks: {len(st.session_state.chunks)}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:var(--secondary-text);">Source: {st.session_state.source_name}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--secondary-text);">No source loaded yet.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3>📂 Source Input</h3>', unsafe_allow_html=True)
        source_type = st.selectbox(
            "Select Source Type:",
            ["Document Upload", "Website URL"],
            key="source_type_select",
            help="Choose between uploading a file or entering a URL"
        )

        if source_type == "Document Upload":
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'docx', 'eml'],
                help="Supported formats: PDF, DOCX, EML",
                key="file_upload"
            )
            if uploaded_file:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if st.button("📄 Process Document", key="process_doc"):
                    success, message = process_source(file_extension, uploaded_file.read(), uploaded_file.name)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        else:
            url = st.text_input("Enter Website URL:", placeholder="https://example.com", key="website_url")
            if url and st.button("🌐 Process Website", key="process_web"):
                if not url.lower().startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                else:
                    success, message = process_source("website", url, url)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3>⚙️ Settings</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color:var(--secondary-text);">Adjust preferences or clear session data here.</p>', unsafe_allow_html=True)
        if st.button("🔄 Clear Session", key="clear_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.success("Session cleared successfully!")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.is_loaded:
        st.markdown('<div class="content">', unsafe_allow_html=True)
        st.success("📊 Source Loaded Successfully!")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('<div class="subheading">💬 Chat with AI</div>', unsafe_allow_html=True)
            question = st.text_input("Ask a Question:", placeholder="Type your question here...", key="question_input")
            if st.button("🔍 Submit", type="primary", disabled=not question.strip(), key="get_answer_btn"):
                if question.strip():
                    with st.spinner("Generating response..."):
                        response = get_ai_response_with_memory(question)
                        st.markdown(f'<div class="chat-message user"><div class="chat-bubble">You: {question}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-message bot"><div class="chat-bubble">Bot: {response["answer"]}</div></div>', unsafe_allow_html=True)
                        conf = response["confidence"]
                        if conf == "high":
                            st.success(f"🎯 High Confidence")
                        elif conf == "medium":
                            st.warning(f"⚡ Medium Confidence")
                        else:
                            st.info(f"💭 Low Confidence")
                        st.info(f"📊 Used {response['sources_used']} source chunks")
                        if response.get("context"):
                            with st.expander("📚 Source Excerpts"):
                                for i, chunk in enumerate(response["context"], 1):
                                    st.text_area(f"Source {i}:", chunk, height=100, disabled=True, key=f"source_{i}")

            if st.session_state.chat_history:
                st.markdown('<div class="subheading">🗨️ Chat History</div>', unsafe_allow_html=True)
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for chat in st.session_state.chat_history[-10:]:
                    st.markdown(f'<div class="chat-message user"><div class="chat-bubble"><b>You:</b> {chat["message"]}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-message bot"><div class="chat-bubble"><b>Bot:</b> {chat["response"]}</div></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="subheading">ℹ️ How to Use</div>', unsafe_allow_html=True)
            st.markdown("""
                - **Upload or Enter URL**: Select a document (PDF, DOCX, EML) or input a website URL.
                - **Process Content**: Click the process button to analyze the content.
                - **Ask Questions**: Use the chat interface to query the AI.
                - **Context Retention**: The AI maintains context for follow-up questions.
                - **View Sources**: Expand to review source excerpts used in responses.
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="content" style="text-align:center;">
                <p style="font-size:1.2rem; color:var(--secondary-text);">
                    Please upload a document or enter a website URL to begin.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <footer>
            <p>&copy; 2025 Intelligentsia - A Proud Initiative of KL University Club. This project is crafted by the Intelligentsia club members as part of our college club activities. Learn more about us at <a href="https://intelligentsiaclub.netlify.app/">/intelligentsia</a> | Contact: intelligentsia@gmail.com</p>

        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()