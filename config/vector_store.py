import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader


def load_pdf():
    file_path = "./documents/pdf/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    return docs


docs = load_pdf()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)

vector_store = Chroma(embedding_function=embedding_model)
ids = vector_store.add_documents(documents=all_splits)
