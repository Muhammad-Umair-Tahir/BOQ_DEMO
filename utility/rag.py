import os
import logging
import asyncio
from typing import Optional

from agno.agent import Agent
from agno.models.google import Gemini
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.embedder.google import GeminiEmbedder
import dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAG_AGENT")

# Load environment variables
dotenv.load_dotenv()

# Global knowledge base instance
_knowledge_base: Optional[PDFKnowledgeBase] = None

async def initialize_rag_components():
    """
    Initialize RAG components asynchronously.
    
    Returns:
        PDFKnowledgeBase: Initialized knowledge base for use in agents
        
    Raises:
        Exception: If initialization fails
    """
    global _knowledge_base
    
    try:
        logger.info("Initializing RAG components...")
        
        # Get API keys and model IDs with fallbacks
        embedder_model = os.getenv("RAG_EMBEDDER_MODEL", "models/text-embedding-004")
        embedder_api_key = os.getenv("RAG_EMBEDDER_MODEL_API_KEY")
        
        if not embedder_api_key:
            logger.warning("RAG_EMBEDDER_MODEL_API_KEY not found in environment")
        
        # Initialize embedder
        logger.debug(f"Initializing GeminiEmbedder with model: {embedder_model}")
        embedder = GeminiEmbedder(
            dimensions=786,
            id=embedder_model, 
            api_key=embedder_api_key
        )
        
        # Initialize ChromaDB - run any potentially blocking operations in thread pool
        logger.debug("Initializing ChromaDB")
        vector_db = await asyncio.to_thread(
            ChromaDb,
            collection="ArchitectureStandards",
            path="data/chromaDB",
            embedder=embedder,
            persistent_client=True
        )
        
        # Initialize knowledge base
        logger.debug("Initializing PDFKnowledgeBase")
        _knowledge_base = await asyncio.to_thread(
            PDFKnowledgeBase,
            vector_db=vector_db
        )
        
        logger.info("RAG components initialized successfully")
        return _knowledge_base
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
        raise

async def rag_agent():
    """
    Get an initialized knowledge base asynchronously.
    
    If the knowledge base hasn't been initialized yet, this will
    initialize it first.
    
    Returns:
        PDFKnowledgeBase: Initialized knowledge base for use in agents
    """
    global _knowledge_base
    
    if _knowledge_base is None:
        logger.info("Knowledge base not initialized, initializing now")
        return await initialize_rag_components()
    
    return _knowledge_base

# For backwards compatibility with synchronous code
def get_rag_agent_sync():
    """
    Synchronous accessor for the knowledge base.
    
    This is for backward compatibility only. If the knowledge base
    isn't initialized, it will return None.
    
    Returns:
        Optional[PDFKnowledgeBase]: The knowledge base if initialized, None otherwise
    """
    return _knowledge_base