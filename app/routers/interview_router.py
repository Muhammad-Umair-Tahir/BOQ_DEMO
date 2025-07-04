from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from agents.interview_agent import InterviewAgent, initialize_shared_resources
import logging

router = APIRouter()

# Configure logging
logger = logging.getLogger("INTERVIEW_ROUTER")

# Initialize as None, will be populated during first request or startup
agent = None

async def init_interview_agent():
    """Initialize the interview agent asynchronously."""
    global agent
    try:
        # Now unpacking three values from initialize_shared_resources
        memory, storage, knowledge = await initialize_shared_resources()
        # Pass all three parameters to the InterviewAgent constructor
        agent = InterviewAgent(memory=memory, storage=storage, knowledge=knowledge)
        logger.info("InterviewAgent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize InterviewAgent: {e}")
        raise

async def collect_streamed_content(data: str, user_id: Optional[str], session_id: Optional[str]) -> str:
    """
    Runs the agent and collects all parts of the response into a single combined string.
    """
    global agent
    if agent is None:
        logger.info("Initializing InterviewAgent on first request")
        agent = await init_interview_agent()
    
    content_parts = []
    # interview() returns an async generator, not a coroutine
    response_iter = agent.interview(data, user_id=user_id, session_id=session_id)
    async for event in response_iter:
        text = None
        if hasattr(event, "content") and event.content:
            text = event.content
        elif hasattr(event, "delta") and event.delta:
            text = event.delta
        if text:
            content_parts.append(text)
    return "".join(content_parts)

@router.post("/interview", tags=["interview"])
async def interview_endpoint(
    data: str = Form(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Endpoint that runs the interview agent and returns the full response once ready.
    """
    try:
        # Validate input
        if not data.strip():
            raise HTTPException(status_code=400, detail="Input data cannot be empty.")
        
        # Debugging logs
        logger.debug(f"Received data: {data[:100]}..." if len(data) > 100 else data)
        logger.debug(f"User ID: {user_id}, Session ID: {session_id}")
        
        # Collect the streamed content from the interview agent
        full_text = await collect_streamed_content(data, user_id, session_id)
        
        # Ensure full_text is not empty before returning
        if not full_text.strip():
            logger.error("Generated interview response is empty.")
            raise HTTPException(status_code=500, detail="Failed to generate interview response.")
            
        # Return the response
        logger.info(f"Successfully generated interview response for user: {user_id}")
        return JSONResponse(content={"content": full_text})
        
    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate interview response: {str(e)}")