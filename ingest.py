import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import json
import shutil
import hashlib

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Import our custom modules
from config import config
class DocumentManager:
    """Manages document operations: add, remove, list, and track changes"""
    
    def __init__(self, data_path: str = "data/", metadata_file: str = "document_metadata.json"):
        self.data_path = Path(data_path)
        self.metadata_file = Path(metadata_file)
        self.metadata = self._load_metadata()
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load document metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save document metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def add_document(self, source_path: str, copy_file: bool = True) -> Tuple[bool, str]:
        """
        Add a document to the data directory
        
        Args:
            source_path: Path to the source document
            copy_file: Whether to copy the file or just track it
            
        Returns:
            Tuple of (success, message)
        """
        try:
            source_path = Path(source_path)
            
            # Validate file exists
            if not source_path.exists():
                return False, f"File not found: {source_path}"
            
            # Validate file extension
            if source_path.suffix.lower() not in ['.pdf', '.txt', '.docx']:
                return False, f"Unsupported file type: {source_path.suffix}"
            
            # Check file size (max 100MB)
            file_size_mb = source_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 100:
                return False, f"File too large: {file_size_mb:.2f}MB (max 100MB)"
            
            # Determine destination path
            dest_path = self.data_path / source_path.name
            
            # Handle duplicate names
            counter = 1
            original_dest = dest_path
            while dest_path.exists():
                stem = original_dest.stem
                suffix = original_dest.suffix
                dest_path = self.data_path / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy file if requested
            if copy_file:
                shutil.copy2(source_path, dest_path)
            else:
                dest_path = source_path
            
            # Calculate file hash
            file_hash = self._get_file_hash(dest_path)
            
            # Update metadata
            self.metadata[str(dest_path)] = {
                'original_path': str(source_path),
                'added_date': datetime.now().isoformat(),
                'file_size': file_size_mb,
                'file_hash': file_hash,
                'processed': False
            }
            
            self._save_metadata()
            
            logger.info(f"Document added: {dest_path}")
            return True, f"Document added successfully: {dest_path.name}"
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False, f"Error adding document: {str(e)}"
    
    def remove_document(self, filename: str, delete_file: bool = True) -> Tuple[bool, str]:
        """
        Remove a document from tracking and optionally delete the file
        
        Args:
            filename: Name of the file to remove
            delete_file: Whether to delete the actual file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            file_path = self.data_path / filename
            
            # Remove from metadata
            if str(file_path) in self.metadata:
                del self.metadata[str(file_path)]
                self._save_metadata()
            
            # Delete file if requested and exists
            if delete_file and file_path.exists():
                file_path.unlink()
                logger.info(f"Document file deleted: {file_path}")
            
            logger.info(f"Document removed from tracking: {filename}")
            return True, f"Document removed successfully: {filename}"
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False, f"Error removing document: {str(e)}"
    
    def list_documents(self) -> List[Dict]:
        """List all tracked documents with their metadata"""
        try:
            documents = []
            for file_path, metadata in self.metadata.items():
                file_path = Path(file_path)
                
                doc_info = {
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size_mb': metadata.get('file_size', 0),
                    'added_date': metadata.get('added_date', 'Unknown'),
                    'processed': metadata.get('processed', False),
                    'exists': file_path.exists()
                }
                documents.append(doc_info)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def get_unprocessed_documents(self) -> List[str]:
        """Get list of documents that haven't been processed yet"""
        try:
            unprocessed = []
            for file_path, metadata in self.metadata.items():
                if not metadata.get('processed', False):
                    if Path(file_path).exists():
                        unprocessed.append(file_path)
            
            return unprocessed
            
        except Exception as e:
            logger.error(f"Error getting unprocessed documents: {e}")
            return []
    
    def mark_as_processed(self, file_path: str):
        """Mark a document as processed"""
        try:
            if file_path in self.metadata:
                self.metadata[file_path]['processed'] = True
                self.metadata[file_path]['processed_date'] = datetime.now().isoformat()
                self._save_metadata()
                logger.info(f"Document marked as processed: {file_path}")
        except Exception as e:
            logger.error(f"Error marking document as processed: {e}")
    
    def check_for_changes(self) -> List[str]:
        """Check if any tracked documents have been modified"""
        try:
            changed_files = []
            
            for file_path, metadata in self.metadata.items():
                file_path = Path(file_path)
                
                if not file_path.exists():
                    continue
                
                # Check if file hash has changed
                current_hash = self._get_file_hash(file_path)
                stored_hash = metadata.get('file_hash', '')
                
                if current_hash != stored_hash:
                    changed_files.append(str(file_path))
                    # Update metadata
                    metadata['file_hash'] = current_hash
                    metadata['processed'] = False  # Mark as needing reprocessing
                    metadata['last_modified'] = datetime.now().isoformat()
            
            if changed_files:
                self._save_metadata()
                logger.info(f"Detected changes in {len(changed_files)} files")
            
            return changed_files
            
        except Exception as e:
            logger.error(f"Error checking for changes: {e}")
            return []
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the document collection"""
        try:
            docs = self.list_documents()
            
            stats = {
                'total_documents': len(docs),
                'processed_documents': sum(1 for doc in docs if doc['processed']),
                'unprocessed_documents': sum(1 for doc in docs if not doc['processed']),
                'total_size_mb': sum(doc['size_mb'] for doc in docs),
                'file_types': {},
                'missing_files': sum(1 for doc in docs if not doc['exists'])
            }
            
            # Count file types
            for doc in docs:
                ext = Path(doc['filename']).suffix.lower()
                stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}
    
    def cleanup_missing_files(self) -> Tuple[int, List[str]]:
        """Remove tracking for files that no longer exist"""
        try:
            removed_files = []
            
            for file_path in list(self.metadata.keys()):
                if not Path(file_path).exists():
                    del self.metadata[file_path]
                    removed_files.append(file_path)
            
            if removed_files:
                self._save_metadata()
                logger.info(f"Cleaned up {len(removed_files)} missing files")
            
            return len(removed_files), removed_files
            
        except Exception as e:
            logger.error(f"Error cleaning up missing files: {e}")
            return 0, []
# Setup logging
config.setup_logging()
logger = logging.getLogger(__name__)

class EnhancedDocumentIngestor:
    """Enhanced document ingestion with error handling and progress tracking"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.doc_manager = DocumentManager(self.config.data_path)
        self.embeddings = None
        self.text_splitter = None
        self._setup_components()
    
    def _setup_components(self):
        """Initialize embeddings and text splitter"""
        try:
            self.embeddings = OllamaEmbeddings()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            logger.info("✅ Components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def _get_loader_for_file(self, file_path: Path):
        """Get appropriate loader based on file extension"""
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return PyPDFLoader(str(file_path))
        elif extension == '.txt':
            return TextLoader(str(file_path), encoding='utf-8')
        elif extension in ['.docx', '.doc']:
            return UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def load_documents(self, file_paths: List[str] = None) -> Tuple[List[Document], List[str]]:
        """
        Load documents from specified paths or data directory
        
        Returns:
            Tuple of (loaded_documents, failed_files)
        """
        loaded_docs = []
        failed_files = []
        
        try:
            if file_paths is None:
                # Load all files from data directory
                file_paths = []
                for ext in self.config.supported_extensions:
                    file_paths.extend(Path(self.config.data_path).glob(f'*{ext}'))
                file_paths = [str(p) for p in file_paths]
            
            if not file_paths:
                logger.warning("No files found to process")
                return [], []
            
            logger.info(f"Loading {len(file_paths)} files...")
            
            for file_path in file_paths:
                try:
                    file_path = Path(file_path)
                    
                    # Check file size
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > self.config.max_file_size_mb:
                        logger.warning(f"⚠️ Skipping large file: {file_path.name} ({file_size_mb:.2f}MB)")
                        failed_files.append(str(file_path))
                        continue
                    
                    # Load document
                    loader = self._get_loader_for_file(file_path)
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source_file': str(file_path),
                            'file_name': file_path.name,
                            'file_size_mb': file_size_mb,
                            'loaded_at': datetime.now().isoformat()
                        })
                    
                    loaded_docs.extend(docs)
                    logger.info(f"✅ Loaded: {file_path.name} ({len(docs)} pages)")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {file_path}: {e}")
                    failed_files.append(str(file_path))
            
            logger.info(f"📚 Successfully loaded {len(loaded_docs)} documents from {len(file_paths) - len(failed_files)} files")
            return loaded_docs, failed_files
            
        except Exception as e:
            logger.error(f"❌ Error in load_documents: {e}")
            return [], file_paths or []
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with progress tracking"""
        try:
            if not documents:
                logger.warning("No documents to split")
                return []
            
            logger.info(f"Splitting {len(documents)} documents into chunks...")
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Apply chunk limit if specified
            if getattr(self.config, "max_chunks", None) is not None:
                original_count = len(chunks)
                chunks = chunks[:self.config.max_chunks]
                logger.info(f"Applied chunk limit: {original_count} → {len(chunks)} chunks")
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'total_chunks': len(chunks)
                })
            
            logger.info(f"✅ Created {len(chunks)} text chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error splitting documents: {e}")
            return []
    
    def create_vector_store(self, chunks: List[Document]) -> Optional[FAISS]:
        """Create FAISS vector store from document chunks"""
        try:
            if not chunks:
                logger.error("No chunks to create vector store")
                return None
            
            logger.info(f"Creating FAISS vector store from {len(chunks)} chunks...")
            
            # Create vector store in batches to handle memory efficiently
            batch_size = 100
            vector_store = None
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                
                if vector_store is None:
                    # Create initial vector store
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    # Add to existing vector store
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    vector_store.merge_from(batch_store)
            
            logger.info("✅ Vector store created successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {e}")
            return None
    
    def save_vector_store(self, vector_store: FAISS) -> bool:
        """Save vector store to disk"""
        try:
            if vector_store is None:
                logger.error("No vector store to save")
                return False
            
            # Create directory if it doesn't exist
            Path(self.config.db_faiss_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save vector store
            vector_store.save_local(self.config.db_faiss_path)
            
            logger.info(f"✅ Vector store saved to {self.config.db_faiss_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving vector store: {e}")
            return False
    
    def process_documents(self, file_paths: List[str] = None) -> Tuple[bool, Dict]:
        """
        Complete document processing pipeline
        
        Returns:
            Tuple of (success, statistics)
        """
        start_time = datetime.now()
        stats = {
            'start_time': start_time.isoformat(),
            'files_processed': 0,
            'files_failed': 0,
            'documents_loaded': 0,
            'chunks_created': 0,
            'processing_time': 0
        }
        
        try:
            # Load documents
            documents, failed_files = self.load_documents(file_paths)
            stats['documents_loaded'] = len(documents)
            stats['files_failed'] = len(failed_files)
            stats['files_processed'] = len(documents) + len(failed_files)
            
            if not documents:
                logger.error("No documents loaded - cannot proceed")
                return False, stats
            
            # Split documents
            chunks = self.split_documents(documents)
            stats['chunks_created'] = len(chunks)
            
            if not chunks:
                logger.error("No chunks created - cannot proceed")
                return False, stats
            
            # Create vector store
            vector_store = self.create_vector_store(chunks)
            if vector_store is None:
                logger.error("Failed to create vector store")
                return False, stats
            
            # Save vector store
            if not self.save_vector_store(vector_store):
                logger.error("Failed to save vector store")
                return False, stats
            
            # Update document manager
            if file_paths:
                for file_path in file_paths:
                    if file_path not in failed_files:
                        self.doc_manager.mark_as_processed(file_path)
            
            # Calculate processing time
            end_time = datetime.now()
            stats['processing_time'] = (end_time - start_time).total_seconds()
            stats['end_time'] = end_time.isoformat()
            
            logger.info(f"🎉 Processing completed successfully in {stats['processing_time']:.2f} seconds")
            return True, stats
            
        except Exception as e:
            logger.error(f"❌ Error in process_documents: {e}")
            stats['error'] = str(e)
            return False, stats
    
    def incremental_update(self) -> Tuple[bool, Dict]:
        """Process only new or modified documents"""
        try:
            logger.info("🔄 Starting incremental update...")
            
            # Check for changes
            changed_files = self.doc_manager.check_for_changes()
            unprocessed_files = self.doc_manager.get_unprocessed_documents()
            
            files_to_process = list(set(changed_files + unprocessed_files))
            
            if not files_to_process:
                logger.info("✅ No new or modified files to process")
                return True, {'files_processed': 0, 'message': 'No updates needed'}
            
            logger.info(f"📄 Processing {len(files_to_process)} files...")
            return self.process_documents(files_to_process)
            
        except Exception as e:
            logger.error(f"❌ Error in incremental update: {e}")
            return False, {'error': str(e)}

def main():
    """Main function to run document ingestion"""
    try:
        logger.info("🚀 Starting Enhanced Document Ingestion")
        
        # Validate configuration
        if not config.validate_paths():
            logger.error("❌ Configuration validation failed")
            sys.exit(1)
        
        # Initialize ingestor
        ingestor = EnhancedDocumentIngestor(config)
        
        # Check for command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == '--incremental':
                success, stats = ingestor.incremental_update()
            else:
                # Process specific files
                file_paths = sys.argv[1:]
                success, stats = ingestor.process_documents(file_paths)
        else:
            # Process all files
            success, stats = ingestor.process_documents()
        
        # Print results
        if success:
            logger.info("Ingestion completed successfully!")
            logger.info(f"Files processed: {stats.get('files_processed', 0)}")
            logger.info(f"Documents loaded: {stats.get('documents_loaded', 0)}")
            logger.info(f"Chunks created: {stats.get('chunks_created', 0)}")
            logger.info(f"Processing time: {stats.get('processing_time', 0):.2f} seconds")
            
            if stats.get('files_failed', 0) > 0:
                logger.warning(f" Files failed: {stats['files_failed']}")
            
            print("\n🎉 Ready to run chatbot with: chainlit run main.py")
        else:
            logger.error("Ingestion failed!")
            if 'error' in stats:
                logger.error(f"Error: {stats['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()