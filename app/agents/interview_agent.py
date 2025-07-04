"""
Interview Agent Module - Conducts architectural client interviews.

This module provides a specialized agent for interviewing architectural clients
to understand their needs, preferences, and project requirements.
"""

import os
import json
import logging
import asyncio
from typing import AsyncIterator, Optional, Dict, Any
from pathlib import Path

import dotenv
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.utils.pprint import pprint_run_response

from utility.utils import shared_memory, shared_storage
from utility.rag import rag_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("INTERVIEW_AGENT")

# Load environment variables
dotenv.load_dotenv()

# Define constants
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_USER = "default_user"
DEFAULT_SESSION = "default_session"
MAX_LOG_CONTENT_LENGTH = 100  # Maximum length for logging content

# Get the absolute path to the JSON file
instructions_path = os.path.join(os.path.dirname(__file__), '..', 'instructions', 'interview_agent.json')

# Load the JSON file with error handling
try:
    with open(instructions_path, 'r', encoding='utf-8') as f:
        INTERVIEW_AGENT_INSTRUCTIONS = json.load(f)
    logger.info("Successfully loaded interview agent instructions")
except FileNotFoundError:
    logger.error(f"Instructions file not found: {instructions_path}")
    # Provide fallback minimal instructions
    INTERVIEW_AGENT_INSTRUCTIONS = {
        "system_message": "Conduct a professional interview for architectural requirements gathering."
    }
    logger.warning("Using fallback instructions")
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse instructions JSON: {e}")
    INTERVIEW_AGENT_INSTRUCTIONS = {
        "system_message": "Conduct a professional interview for architectural requirements gathering."
    }
    logger.warning("Using fallback instructions")

async def initialize_shared_resources():
    """
    Initialize shared memory and storage asynchronously.
    
    Returns:
        Tuple: (memory, storage, knowledge) instances for agent initialization
        
    Raises:
        Exception: If resource initialization fails
    """
    try:
        logger.info("Initializing shared memory, storage, and knowledge base")
        memory = await shared_memory()
        storage = await shared_storage()
        knowledge = await rag_agent()  # Now properly awaiting the async rag_agent
        logger.info("Shared resources initialized successfully")
        return memory, storage, knowledge
    except Exception as e:
        logger.error(f"Failed to initialize shared resources: {e}")
        raise

class InterviewAgent(Agent):
    """
    An agent specialized in conducting architectural client interviews.
    
    This agent is designed to engage in professional conversations with clients
    to understand their architectural needs, preferences, and project requirements.
    """
    
    def __init__(self, memory=None, storage=None, knowledge=None):
        """
        Initialize the InterviewAgent with shared resources.
        
        Args:
            memory: Shared memory instance for conversation history
            storage: Shared storage instance for persistent data
            knowledge: Knowledge base for domain-specific information
            
        Raises:
            ValueError: If required environment variables are missing
        """
        self.memory = memory
        self.storage = storage
        
        # Get API key and model from environment variables with fallbacks
        api_key = os.getenv("INTERVIEW_AGENT_API_KEY")
        if not api_key:
            logger.warning("INTERVIEW_AGENT_API_KEY not found in environment")
            
        model_id = os.getenv("INTERVIEW_AGENT_MODEL", DEFAULT_MODEL)
        logger.info(f"Using model: {model_id}")
        
        # Initialize the base Agent class with our configuration
        super().__init__(
            name="InterviewAgent",
            agent_id="interview_agent",
            model=Gemini(id=model_id, api_key=api_key),
            memory=memory,
            storage=storage,
            description=(
                "You are a professional interviewer for an architecture firm whose main purpose is to conduct "
                "client interviews to understand their needs and preferences. Clients can be individuals or "
                "companies, from family persons to business owners. You are trained to ask the right questions "
                "and extract valuable information to inform the design process."
            ),
            instructions=INTERVIEW_AGENT_INSTRUCTIONS,
            knowledge=knowledge,  # Now passing the pre-initialized knowledge
            search_knowledge=True,
            enable_session_summaries=True,
            
            # CHAT HISTORY CONFIG
            add_history_to_messages=True,
            num_history_runs=3,
            read_chat_history=True,
            
            # MEMORY CONFIG
            debug_mode=True,
        )
        logger.info("InterviewAgent initialized successfully")

    async def interview(
        self, 
        data: str, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None
    ) -> AsyncIterator[RunResponse]:
        """
        Conduct an interview to gather architectural design requirements.
        
        Args:
            data: User input or conversation data
            user_id: User identifier for session management
            session_id: Session identifier for conversation continuity
            
        Returns:
            AsyncIterator[RunResponse]: Streaming response events with interview questions/responses
            
        Raises:
            ValueError: If input data is empty
            Exception: For any errors during interview process
        """
        # Validate input parameters
        if not data or not data.strip():
            logger.error("Empty input data provided to interview method")
            raise ValueError("Input data cannot be empty")
            
        # Use default values if not provided
        user_id = user_id or DEFAULT_USER
        session_id = session_id or DEFAULT_SESSION
        
        # Truncate long data in logs
        log_data = data[:MAX_LOG_CONTENT_LENGTH] + "..." if len(data) > MAX_LOG_CONTENT_LENGTH else data
        logger.info(f"Conducting interview with data: {log_data}")
        logger.info(f"User ID: {user_id}, Session ID: {session_id}")
        
        try:
            # Conduct interview using the agent's run method
            logger.debug("Starting interview")
            response_stream = await self.arun(
                data, 
                user_id=user_id, 
                session_id=session_id, 
                stream=True
            )
            
            # Yield each response event
            async for event in response_stream:
                logger.debug(f"Event received: {type(event)}")
                yield event
                
            logger.info("Interview completed successfully")
            
        except Exception as e:
            logger.error(f"Interview failed: {e}", exc_info=True)
            raise

async def main():
    """
    Async main function for testing the Interview agent as a standalone CLI application.
    
    This function provides an interactive CLI interface for testing the interview agent
    without requiring the full web API.
    """
    try:
        # Initialize shared resources
        logger.info("Starting Interview Agent CLI")
        memory, storage, knowledge = await initialize_shared_resources()  # Now unpacking three values
        agent = InterviewAgent(memory=memory, storage=storage, knowledge=knowledge)  # Pass knowledge to agent

        print("\n" + "=" * 50)
        print("🏗️  Interview Agent - Architectural Requirements")
        print("=" * 50)
        print("Start your architectural project interview (or type 'exit' to quit)")
        print("-" * 50)
        
        # Optional initial greeting
        print("\n🤖 Assistant:")
        print("-" * 30)
        print("Hello! I'm your architectural interviewer today. I'll help gather information about your project needs and preferences. What kind of architectural project are you looking to start?")
        print("\n" + "-" * 30)
        print()
        
        while True:
            try:
                # Use asyncio-compatible input
                text = await asyncio.to_thread(input, "You: ")
                
                if text.lower() in ('exit', 'quit'):
                    print("Thank you for using the Interview Agent. Goodbye!")
                    break

                if not text.strip():
                    print("Please enter your message.")
                    continue
                
                print(f"\n🤖 Assistant:")
                print("-" * 30)

                # Conduct interview asynchronously
                try:
                    response_generator = agent.interview(
                        text, 
                        user_id="test_user", 
                        session_id="test_session"
                    )
                    
                    events = []
                    async for event in response_generator:
                        events.append(event)
                        
                    if events:
                        pprint_run_response(events, markdown=True)
                    else:
                        print("No response received from the agent.")
                        
                except ValueError as ve:
                    print(f"❌ Input error: {ve}")
                    continue

                print("\n" + "-" * 30)
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterview cancelled by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.exception("Error in main loop")
                print("Please try again.\n")
                
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        print(f"\n❌ Critical error: {e}")
        print("The application will now exit.")
        return 1
        
    return 0

if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    exit(exit_code)