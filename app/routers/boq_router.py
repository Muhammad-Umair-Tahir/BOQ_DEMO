from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from agents.boq_agent import BOQAgent, initialize_shared_resources

# Configure logging
logger = logging.getLogger("BOQ_ROUTER")

router = APIRouter()

# Global variable to hold the agent instance
boq_agent = None

async def init_boq_agent():
    """Initialize the BOQ agent asynchronously."""
    global boq_agent
    try:
        # Now unpacking three values from initialize_shared_resources
        memory, storage, knowledge = await initialize_shared_resources()
        # Pass all three parameters to the BOQAgent constructor
        boq_agent = BOQAgent(memory=memory, storage=storage, knowledge=knowledge)
        logger.info("BOQAgent initialized successfully")
        return boq_agent
    except Exception as e:
        logger.error(f"Failed to initialize BOQAgent: {e}")
        raise

async def collect_streamed_content(data: str, user_id: Optional[str], session_id: Optional[str]) -> str:
    """
    Runs the BOQ agent and collects all parts of the response into a single combined string.
    """
    global boq_agent
    if boq_agent is None:
        logger.info("Initializing BOQAgent on first request")
        boq_agent = await init_boq_agent()
        
    content_parts = []
    response_iter = boq_agent.generate_boq(data, user_id=user_id, session_id=session_id)
    async for event in response_iter:
        text = None
        if hasattr(event, "content") and event.content:
            text = event.content
        elif hasattr(event, "delta") and event.delta:
            text = event.delta
        if text:
            content_parts.append(text)
    return "".join(content_parts)

@router.post("/generate-boq", tags=["boq"])
async def generate_boq_endpoint(
    data: str = Form(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Endpoint that runs the BOQ agent and returns the full response once ready.
    """
    try:
        # Validate input
        if not data.strip():
            raise HTTPException(status_code=400, detail="Project data cannot be empty.")
        if not user_id or not user_id.strip():
            raise HTTPException(status_code=400, detail="User ID cannot be empty.")
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

        # Debugging logs
        logger.debug(f"Received data: {data[:100]}..." if len(data) > 100 else data)
        logger.debug(f"User ID: {user_id}, Session ID: {session_id}")

        # Collect the streamed content from the BOQ agent
        full_boq = await collect_streamed_content(data, user_id, session_id)

        # Ensure full_boq is not empty before returning
        if not full_boq.strip():
            logger.error("Generated BoQ is empty.")
            raise HTTPException(status_code=500, detail="Failed to generate BoQ.")

        # Return the response
        logger.info(f"Successfully generated BoQ for user: {user_id}, session: {session_id}")
        return JSONResponse(content={"boq": full_boq})

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate BoQ: {str(e)}")