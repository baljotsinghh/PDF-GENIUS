import streamlit as st
from pdf_handler import PDFTextExtractor
from chunker import TextChunker
from embedder import EmbeddingManager
from vector_base import VectorStore
from gemini_wrapper import GeminiAPI
from chat_memory import ChatMemoryManager
import torch
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

def main():
    st.set_page_config(page_title="Smart PDF Chatbot", layout="centered")

    # Sidebar customization
    st.sidebar.title("⚙️ Settings")
    user_name = st.sidebar.text_input("Enter your name", "User")
    use_mistral = st.sidebar.radio("Choose LLM", ["LLM 1 (By default)", "Add"])
    st.sidebar.link_button("Visit Syllabus Scraper", "https://cse-syllabus-scrapper.streamlit.app/")
    if st.sidebar.button("🧹 Reset Chat Memory"):
        st.session_state.chat_history = []

    st.title("📄 Smart PDF Chatbot with Memory")
    st.markdown("Ask intelligent questions from your uploaded PDF with contextual memory 🔁")

    uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

    if uploaded_file:
        extractor = PDFTextExtractor(uploaded_file)
        raw_text = extractor.extract_text()

        if raw_text.strip() == "":
            st.warning("⚠️ No text found in PDF.")
            return

        with st.expander("🔍 Raw Text Preview"):
            st.text(raw_text[:700] + "...")

        chunker = TextChunker(raw_text, chunk_size=500, overlap=50)
        chunks = chunker.chunk_text()

        st.success(f"✅ Extracted and chunked into {len(chunks)} pieces")

        with st.spinner("🔐 Generating Embeddings..."):
            embedder = EmbeddingManager()
            embeddings = embedder.encode_chunks(chunks)

        vs = VectorStore(dim=embeddings.shape[1])
        vs.add_embeddings(embeddings)

        st.success("💾 Embeddings stored in vector DB (FAISS)")

        memory = ChatMemoryManager(st.session_state)

        st.header("🗣️ Interactive Chat Interface")

        # Initialize memory object
        memory = ChatMemoryManager(st.session_state)

        # Display chat history using new Streamlit chat format
        for turn in memory.get_chat_history():
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.markdown(turn["content"])

        # Input widget using new chat format
        user_query = st.chat_input("💬 Type your question...")

        if user_query:
            memory.add_user_message(user_query)
            with st.chat_message("user"):
                st.markdown(user_query)

            # Embedding & retrieval
            query_emb = embedder.encode_chunks([user_query])
            indices, _ = vs.search(query_emb, top_k=3)
            relevant_chunks = [chunks[i] for i in indices[0]]

            # Context for prompt
            context = memory.get_formatted_context()
            full_prompt = f"{context}\n\nRelevant Info:\n{relevant_chunks}\n\n{user_name}: {user_query}"

            # Response from LLM
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        llm = GeminiAPI()
                        response = llm.ask(relevant_chunks, full_prompt)
                        memory.add_bot_response(response)
                        st.markdown(response)

                    except Exception as e:
                        st.error(f"LLM error: {e}")
                        

if __name__ == "__main__":
    main()