import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# -------- LangChain Imports (FIXED) --------
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint

# -------- Constants --------
FAISS_DB_PATH = "faiss_index"


def get_llm(api_key):
    """
    Uses the NEW HuggingFaceEndpoint API
    (No .post() issue, fully compatible)
    """
    return HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=api_key,
        temperature=0.3,
        max_new_tokens=512
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")

    # -------- Session State --------
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -------- UI --------
    st.title("📚 Chat with your PDFs")
    st.write("Upload PDFs and ask questions")

    # -------- API KEY --------
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_api_key:
        st.error("❌ Add HUGGINGFACEHUB_API_TOKEN in .env file")
        st.stop()

    # -------- Load Existing FAISS --------
    if os.path.exists(FAISS_DB_PATH) and st.session_state.conversation is None:
        with st.spinner("Loading existing knowledge base..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.load_local(
                FAISS_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                llm=get_llm(hf_api_key),
                retriever=vectorstore.as_retriever(),
                memory=memory
            )

    # -------- Sidebar: Upload PDFs --------
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_files = st.file_uploader(
            "Upload PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not pdf_files:
                st.warning("Upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = ""

                    # Read PDFs
                    for pdf in pdf_files:
                        reader = PdfReader(pdf)
                        for page in reader.pages:
                            raw_text += page.extract_text() or ""

                    # Split text
                    splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_text(raw_text)

                    # Create embeddings + FAISS
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    vectorstore = FAISS.from_texts(chunks, embeddings)
                    vectorstore.save_local(FAISS_DB_PATH)

                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )

                    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                        llm=get_llm(hf_api_key),
                        retriever=vectorstore.as_retriever(),
                        memory=memory
                    )

                    st.success("Documents processed successfully")

    # -------- Chat --------
    if st.session_state.conversation:
        question = st.text_input("Ask a question")
        if question:
            response = st.session_state.conversation({"question": question})
            st.session_state.chat_history = response["chat_history"]

    # -------- Display Chat --------
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"**🧑 You:** {msg.content}")
        else:
            st.markdown(f"**🤖 Bot:** {msg.content}")


if __name__ == "__main__":
    main()
