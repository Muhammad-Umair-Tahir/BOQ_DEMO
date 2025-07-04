import os
import logging
import dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from datetime import datetime
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from agno.app.fastapi.app import FastAPIApp
from agents.visualizer_agent import VisualizerAgent as VA, initialize_shared_resources as init_vis_resources
from agents.interview_agent import InterviewAgent as IA, initialize_shared_resources as init_int_resources
from routers.visualizer_router import router as visualizer_router, init_visualizer_agent
from routers.interview_router import router as interview_router, init_interview_agent
from routers.boq_router import router as boq_router, init_boq_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MAIN_APP")

# Load environment variables
dotenv.load_dotenv()

# Create FastAPI app first without agents
app = FastAPI(
    title="VIAB Platform",
    description="Visualization and Interview Agent for Building design",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables for agents
visualizer_agent = None
interview_agent = None
fastapi_app = None

@app.on_event("startup")
async def startup_event():
    """Initialize all agents and FastAPIApp on startup."""
    global visualizer_agent, interview_agent, fastapi_app
    
    logger.info("Starting application initialization...")
    
    try:
        # Initialize all router agents first
        logger.info("Initializing router agents...")
        await init_interview_agent()
        await init_boq_agent()
        await init_visualizer_agent()
        logger.info("Router agents initialized successfully")
        
        # Create FastAPIApp agents with properly initialized resources
        logger.info("Initializing FastAPIApp agents...")
        vis_memory, vis_storage, vis_knowledge = await init_vis_resources()
        int_memory, int_storage, int_knowledge = await init_int_resources()
        
        # Create agents with all required parameters
        visualizer_agent = VA(
            memory=vis_memory, 
            storage=vis_storage, 
            knowledge=vis_knowledge
        )
        
        interview_agent = IA(
            memory=int_memory, 
            storage=int_storage, 
            knowledge=int_knowledge
        )
        
        logger.info("FastAPIApp agents initialized successfully")
        
        # Create the FastAPIApp with properly initialized agents
        fastapi_app = FastAPIApp(
            agents=[visualizer_agent, interview_agent],
            name="VIAB",
            app_id="viab_123",
            description="VIAB: visualization and interview agents integrated."
        )
        
        # Register FastAPIApp routes with our app
        fastapi_routes = fastapi_app.get_app().routes
        for route in fastapi_routes:
            app.routes.append(route)
        
        logger.info("All agents and routes initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        # Continue startup but log the error - the routes may still work
        # Alternatively, you could raise to prevent app from starting with:
        # raise e

# Include your agent routers
app.include_router(interview_router, prefix="/api", tags=["interview"])
app.include_router(boq_router, prefix="/api", tags=["boq"])
app.include_router(visualizer_router, prefix="/api", tags=["visualization"])

# Serve the HTML homepage
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(
        "chatbot_interface.html",
        {
            "request": request,
            "title": "Welcome to VIAB Platform",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when shutting down."""
    logger.info("Application shutting down...")
    # Add any cleanup code here if needed

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)