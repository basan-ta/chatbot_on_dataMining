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
    
    def load_documents(self, file_paths: List[str] = None) -> Tuple[List[Document], List[str]]:
        #load documents from the specified file paths or data directory and returns tupleof(loaded_documents, failed_diles)
        loaded_docs = []
        failed_files = []
        
        try:
            if file_paths is None:
                #load all files from the data directory
                files_paths = []
                for ext in self.config.supported_extensions:
                    files_paths.extend(Path(self.config.data_path).glob(f'*[ext]'))
                
                file_paths = [str(p) for p in files_paths if p.is_file()]

                if not file_paths:
                    logger.warning("No files found to process.")
                    return [], []
                logger.info(f"loadin {len(file_pahts)} files from {self.config.data_path}   directory.")

                for file_path in file_paths:
                    try:
                        file_path = Path(file_path)
                        
                        #check the size of the file 
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)  # Convert bytes to MB
                        if file_size_mb > self.config.max_file_size_mb:
                            logger.warning(f"skkiping large file :{file_path.name} ({file_size_mb: 2f}MB)")
                            failed_files.append(str(file_path))
                            continue
                        #load the document using the appropriate loader
                        loader = self._get_loader_for_file(file_path)
                        docs = loader.load()

                        #adding metadata to the documents
                        for doc in docs:
                            doc.metadata.update({
                                'source_file': str(file_path),,
                                'file_name' : file_path.name,
                                'file_size_mb': file_size_mb,
                                'loaded_at': datatime.now().isoformat()
                            })

                        loaded_docs.extend(docs)
                        logger.info(f"Loaded {file_path.name} ({len(docs)} pages).)")

                    except Exception as e:
                        logger.error(f"Error loading file {file_path}: {e} ")
                        failed_fiels.append(str(file_path))
                    
                    logger.info(f"Successfully loaded {len(loaded_docs)} documents from {len(file_paths)}")
                    return loaded_docs, failed_files
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return [], file_paths or []
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        #split the documents into smaller chinks with progress tracking 
        try:
            if not documents:
                logger.warning("No documents to split")
                return []
            logger.info(f"Splitting {le(documents)} documents into chunks.....")

            #split documents 
            chunks = self.text_splitter.split_documents(documents)

            #apply chunk limit if specified 
            if self.config.max_chunks is not None:
                original_count = len(chunks)
                chunks = chunks[:self.config.max_chunks]
                logger.info(f"Applied chunk limit:{original_count}-> {len(chunks)} chunks")

            #adding metadata on chunks 
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id':i,
                    'chunk_size': len(chunk.page_count)
                    'toatl_chunks': len(chunks)
                })   
            logger.info(f"Created {len(chunks)} text chunks ")
            return chunks 
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
        
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        #create a vector store from the documents and return it
        try:
            if not chunks:
                logger.error("No chunks to create vector store ")
                return None 
            logger.info(f"Creating FAISS vector store from {len(chunks)} chunks ")

            #create the vector store in batches to handle large datasets and memory efficiency
            batch_size = 100
            vector_store = None

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i// batch_size + 1}/{(len(chunks) + batch_size -1)// batch_size}")


                if vector_store is None:
                    #initialize the vector store with the first batch
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    #add the batch to the existing vector store
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_store)
            logger.info("Vector store created successfully.")
            return vector_store 
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
        
    def save_vector_store(self, vector_store: FAISS) -> bool:
        #save vector store to the specified directory
        try:
            if vector_store is None:
                logger.error("No vector store to save.")
                return False 
            
            #create the director if it doesn't exist 
            Path(self.config.db_faiss_path).parent.mkdir(parents=True, exist_ok = True)

            #save the vector store 
            vector_store.save_local(self.congif.db_faiss_path)

            logger.info(f"Vector store saved to {self.config.db_faiss_path}")
            return True 
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            return False
        
    