import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chainlit as cl

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DB_FAISS_PATH = "vectorstores/db_faiss"

# Custom prompt template
custom_prompt_template = """Use the following pieces of information (i.e the context provided) to answer the given question.
If you don't know the answer, please just say that you don't know the answer, please don't try to make up the answer.

Context: {context}

Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def load_llm():
    """
    Load the Ollama LLM model
    """
    try:
        llm = OllamaLLM(model="llama2", temperature=0.1)
        logger.info("✅ Ollama LLM loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"❌ Error loading Ollama LLM: {str(e)}")
        raise

def create_retrieval_chain(llm, prompt, db):
    """
    Create the retrieval QA chain
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def load_qa_bot():
    """
    Load the complete QA bot
    """
    try:
        # Check if FAISS database exists
        if not os.path.exists(DB_FAISS_PATH):
            logger.error(f"❌ FAISS database not found at {DB_FAISS_PATH}")
            logger.error("Please run 'python ingest.py' first to create the database")
            return None
        
        # Load embeddings and database - FIX: Add model parameter
        embeddings = OllamaEmbeddings(model="llama2")
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("✅ FAISS database loaded successfully")
        
        # Load LLM
        llm = load_llm()
        
        # Create prompt
        qa_prompt = set_custom_prompt()
        
        # Create QA chain
        qa_chain = create_retrieval_chain(llm, qa_prompt, db)
        
        logger.info("✅ QA bot initialized successfully")
        return qa_chain
        
    except Exception as e:
        logger.error(f"❌ Error loading QA bot: {str(e)}")
        return None

def format_response(response):
    """
    Format the response with sources
    """
    try:
        result = response.get('result', 'No answer found')
        sources = response.get('source_documents', [])
        
        formatted_response = result
        
        if sources:
            formatted_response += "\n\n📚 **Sources:**\n"
            for i, source in enumerate(sources, 1):
                # Extract metadata
                metadata = source.metadata
                source_info = f"Document {i}"
                
                if 'source' in metadata:
                    filename = os.path.basename(metadata['source'])
                    source_info = f"📄 {filename}"
                
                if 'page' in metadata:
                    source_info += f" (Page {metadata['page'] + 1})"
                
                formatted_response += f"\n{source_info}"
        else:
            formatted_response += "\n\n⚠️ No sources found"
            
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return "Error formatting response"

@cl.on_chat_start
async def start():
    """
    Initialize the chatbot when chat starts
    """
    try:
        # Show loading message
        msg = cl.Message(content="🤖 Initializing DataMining Assistant...")
        await msg.send()
        
        # Load the QA bot
        chain = load_qa_bot()
        
        if chain is None:
            # FIX: Use proper update method
            await msg.update(content="❌ Failed to initialize the bot. Please check the logs and ensure you've run 'python enhanced_ingest.py' first.")
            return
        
        # Update message - FIX: Remove the incorrect line
        await msg.update(content="✅ DataMining Assistant is ready! Ask me anything about your documents.")
        
        # Store the chain in user session
        cl.user_session.set("chain", chain)
        
        logger.info("Chat session started successfully")
        
    except Exception as e:
        logger.error(f"Error starting chat: {str(e)}")
        await cl.Message(content="❌ Error initializing the assistant. Please check the logs.").send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages
    """
    try:
        # Get the chain from user session
        chain = cl.user_session.get("chain")
        
        if chain is None:
            await cl.Message(content="❌ Bot not initialized. Please restart the chat.").send()
            return
        
        # Show thinking message
        thinking_msg = cl.Message(content="🤔 Thinking...")
        await thinking_msg.send()
        
        # Set up callback handler
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True
        
        # Get response from chain
        response = await chain.acall(
            message.content,
            callbacks=[cb]
        )
        
        # Format and send response
        formatted_response = format_response(response)
        
        # Update the thinking message with the response
        await thinking_msg.update(content=formatted_response)
        
        logger.info(f"Successfully processed query: {message.content[:50]}...")
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await cl.Message(content="❌ Error processing your query. Please try again.").send()

if __name__ == "__main__":
    # This won't be called when using 'chainlit run main.py'
    # But useful for testing
    logger.info("Use 'chainlit run main.py' to start the chatbot")