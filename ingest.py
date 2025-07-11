import os 
import sys
import logging
from pathlib import Path
from typing import List, Optimal , Tuple
from datetime import datetime 

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#configuration
DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"

class EnhanceDocumentIngestor:
    #adding error handling and progress tracking 

    def __init__(self, config_obj= None):
        self.config = config_obj
        self.doc_manager = DocumentManager(self.config.data_path)
        self.embeddings = None
        self.text_splitter = None
        self._setup_components()

    def _setup_components(self):
        #initialize embeddings and text splitter 
        try:
            self.embeddings = OllamaEmbeddings(model="llama2")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap = self.config.chunk_overlap,
                length_function=len
                separators=["\n\n", "\n", " ", ".","!", "?", ",", ";"]
            )
            logger.info("Components initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing components : {e}")
            raise
            
    #document loader 

    def __get_loader_for_file(self, file_path: DATA_DIR):
    #Get loader based on the extension of the avilable file 
    extension = file_path.suffix.lower()
    if extension = '.pdf':
        return PyPDFLoader(str(file_path))
    elif extension = '.txt':
        return TextLoader(str(file_path), encoding='utf-8')
    elif extension in ['.docx', '.doc']:
        return UnstructuredWordDocumentLoader(str(file_path))
    else:
        return ValueError(f"unsupported file type : {extension}")
    
    

#create vector database 
def create_vector_database():
    try:
        #check if data directory exists
        if not os.path.exists(DATA_DIR):
            logger.error(f"Data directory '{DATA_DIR}' does not exist.")
            return False 
        
        #check if any pdf files exist in the data directory
        pdf_files = [f for f in os.listdir(DATA_DIR)if f.edndswith('.pdf')]
        if not pdf_files:
            logger.error("no pdf files found in the data directory.")
            return False 
