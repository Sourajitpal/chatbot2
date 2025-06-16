import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import openai

# Groq API setup 
openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_base = "https://api.groq.com/openai/v1"

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with Your PDF")

# Session State Setup 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Upload PDF 
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf and not st.session_state.pdf_processed:
    # Read text from PDF
    pdf_reader = PdfReader(pdf)
    raw_text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            raw_text += page_text

    # Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    # Embedding and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    st.session_state.knowledge_base = knowledge_base
    st.session_state.pdf_processed = True
    st.success("✅ PDF processed and ready for chat!")

# Display previous chat 
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input 
user_input = st.chat_input("Ask something about the PDF...")

if user_input and st.session_state.knowledge_base:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Find relevant chunks
    relevant_docs = st.session_state.knowledge_base.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a helpful assistant.

Use the following context to answer the user's question.

Context:
{context}

Question: {user_input}
Answer:
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = openai.ChatCompletion.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=512
                )
                answer = response['choices'][0]['message']['content'].strip()
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Generation error: {e}")