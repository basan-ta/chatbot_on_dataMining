from lagchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from lagchain_community.embeddings import OllamaEmbeddings
from lagchain_text_splitters import RecursiveCharacterTextSplitter
from lagchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vector_stores/db_faiss"

def create_vector_db():
    # Load documents from the directory
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings for the document chunks
    embeddings = OllamaEmbeddings(model="llama2")

    # Create a FAISS vector store from the document chunks and embeddings
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # Save the vector store to disk
    vector_store.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
    print("Vector database created and saved successfully.")
else:
    print("Ingest module imported successfully.")
    # This allows the module to be imported without executing the main function
    # Useful for testing or when integrating into a larger application