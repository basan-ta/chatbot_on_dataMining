import os
import json
from dataclasses import dataclass
from typing import List, Optional
import logging

@dataclass
class ChatbotConfig:
    """Configuration class for the DataMining Chatbot"""
    
    # Paths
    data_path: str = "data/"
    db_faiss_path: str = "vectorstores/db_faiss"
    config_file: str = "config.json"
    
    # Text Processing
    chunk_size: int = 500
    chunk_overlap: int = 100
    max_chunks: Optional[int] = None  # None means no limit
    
    # Retrieval Settings
    retrieval_k: int = 3  # Number of documents to retrieve
    similarity_threshold: float = 0.7
    
    # LLM Settings
    llm_model: str = "llama2"
    llm_temperature: float = 0.1
    
    # File Processing
    supported_extensions: List[str] = None
    max_file_size_mb: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "chatbot.log"
    
    # UI Settings
    chainlit_port: int = 3000
    welcome_message: str = "Welcome to DataMining Assistant! 🤖📚"
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.pdf', '.txt', '.docx']
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json") -> 'ChatbotConfig':
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                return cls(**config_data)
            else:
                # Create default config file
                default_config = cls()
                default_config.save_to_file(config_path)
                return default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return cls()  # Return default config
    
    def save_to_file(self, config_path: str = None):
        """Save configuration to JSON file"""
        if config_path is None:
            config_path = self.config_file
        
        try:
            config_dict = {
                'data_path': self.data_path,
                'db_faiss_path': self.db_faiss_path,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'max_chunks': self.max_chunks,
                'retrieval_k': self.retrieval_k,
                'similarity_threshold': self.similarity_threshold,
                'llm_model': self.llm_model,
                'llm_temperature': self.llm_temperature,
                'supported_extensions': self.supported_extensions,
                'max_file_size_mb': self.max_file_size_mb,
                'log_level': self.log_level,
                'log_file': self.log_file,
                'chainlit_port': self.chainlit_port,
                'welcome_message': self.welcome_message
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logging.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist or can be created"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_path, exist_ok=True)
            
            # Create vectorstore directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_faiss_path), exist_ok=True)
            
            return True
        except Exception as e:
            logging.error(f"Error validating paths: {e}")
            return False
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )

# Global config instance
config = ChatbotConfig.load_from_file()