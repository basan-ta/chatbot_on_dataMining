# chatbot_on_dataMining

# RAG System - Document-Based AI Assistant

A Retrieval-Augmented Generation (RAG) system that allows users to interact with PDF documents through a conversational interface. The system processes PDF documents, creates embeddings, and enables natural language queries against the document content.

## Features

- PDF Document Processing: Extract and process text from PDF files
- Vector Search: Efficient similarity search using FAISS
- Conversational Interface: Clean chat based UI powered by Chainlit
- Document Retrieval: Find relevant document sections based on user queries
- AI-Powered Responses: Generate contextual answers using retrieved information

## Tech Stack

- pypdf: PDF text extraction and manipulation
- langchain: Framework for LLM application development
- faiss cpu: Vector similarity search and clustering
- langchain community: Additional LangChain integrations
- chainlit: Conversational AI interface

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone <https://github.com/basan-ta/chatbot_on_dataMining.git>
cd rag-system
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install pypdf langchain faiss-cpu langchain_community chainlit
```

4. Set up environment variables:

```bash
# Create a .env file and add your API keys
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
# Or other LLM provider keys as needed
```

## Project Structure

```
Chatbot_on_datamining/
├── main.py                 # Main Chainlit application
├── document_processor.py  # PDF processing and text extraction
├── vector_store.py       # FAISS vector store management
├── retriever.py          # Document retrieval logic
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── documents/            # Directory for PDF files
├── vector_db/           # FAISS index storage
└── README.md            # This file
```

## Usage

### Running the Application

1. Start the Chainlit server:

```bash
chainlit run app.py
```

2. Open your browser and navigate to `http://localhost:3000`

3. Upload PDF documents or place them in the `documents/` folder

4. Start asking questions about your documents!

### Example Queries

- "What is the main topic of this document?"
- "Summarize the key findings in chapter 3"
- "What are the recommendations mentioned?"
- "Find information about [specific topic]"

## Configuration

Edit `config.py` to customize:

- Chunk size: Text splitting parameters
- Overlap: Chunk overlap for better context
- Embeddings model: Choose your preferred embedding model
- LLM settings: Configure your language model
- Vector store: FAISS index parameters

```python
# Example configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
```

## How It Works

1. Document Ingestion: PDFs are processed using pypdf to extract text
2. Text Splitting: Documents are chunked into manageable pieces
3. Embeddings: Text chunks are converted to vector embeddings
4. Vector Storage: Embeddings are stored in FAISS for fast retrieval
5. Query Processing: User questions are embedded and matched against stored vectors
6. Response Generation: Retrieved context is used to generate relevant answers

## API Reference

### Main Components

#### DocumentProcessor

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
documents = processor.load_pdfs("documents/")
chunks = processor.split_text(documents)
```

#### VectorStore

```python
from vector_store import VectorStore

vector_store = VectorStore()
vector_store.add_documents(chunks)
vector_store.save_index("vector_db/")
```

#### Retriever

```python
from retriever import Retriever

retriever = Retriever(vector_store)
relevant_docs = retriever.get_relevant_documents(query)
```

## Troubleshooting

### Common Issues

1. PDF Processing Errors\*\*

   - Ensure PDFs are not password-protected
   - Check file permissions
   - Verify PDF format compatibility

2. Memory Issues\*\*

   - Reduce chunk size in configuration
   - Process documents in smaller batches
   - Consider using faiss-gpu for better performance

3. API Key Issues
   - Verify environment variables are set correctly
   - Check API key permissions and quotas

### Performance Tips

- Use smaller chunk sizes for better precision
- Implement caching for frequently accessed documents
- Consider using GPU acceleration with faiss-gpu
- Optimize embedding model selection based on your use case

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:

- Create an issue on GitHub
- Check the documentation for common solutions
- Review the troubleshooting section above

## Roadmap

- [ ] Support for additional document formats (DOCX, TXT)
- [ ] Multi-language support
- [ ] Advanced filtering and search options
- [ ] Document summarization features
- [ ] Integration with cloud storage providers
- [ ] User authentication and document management

---

Built with ❤️ using LangChain, FAISS, and Chainlit
