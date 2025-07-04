from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
import logging
import os
import tempfile
import asyncio
from uuid import uuid4

from agents.visualizer_agent import VisualizerAgent, initialize_shared_resources

router = APIRouter()

# Configure logging
logger = logging.getLogger("VISUALIZER_ROUTER")

# Create a global variable to hold the agent instance
visualizer_agent = None

class VisualizerResponse(BaseModel):
    content: str

async def process_file(file: UploadFile) -> bytes:
    """Process an uploaded file and return its bytes."""
    try:
        contents = await file.read()
        await file.seek(0)  # Reset file pointer for potential reuse
        return contents
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

async def init_visualizer_agent():
    """Initialize the visualizer agent asynchronously."""
    global visualizer_agent
    try:
        # Now unpacking three values from initialize_shared_resources
        memory, storage, knowledge = await initialize_shared_resources()
        # Pass all three parameters to the VisualizerAgent constructor
        visualizer_agent = VisualizerAgent(memory=memory, storage=storage, knowledge=knowledge)
        logger.info("VisualizerAgent initialized successfully")
        return visualizer_agent
    except Exception as e:
        logger.error(f"Failed to initialize VisualizerAgent: {e}")
        raise

def cleanup_temp_files(file_paths: List[str]):
    """Delete temporary files in background task."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")

async def save_uploaded_files(files: List[UploadFile]) -> List[Tuple[str, str]]:
    """
    Save uploaded files to disk with preserved original filenames.
    
    Args:
        files: List of uploaded files
        
    Returns:
        List of tuples (temp_path, original_filename)
    """
    result = []
    temp_dir = tempfile.gettempdir()
    
    for file in files:
        try:
            # Get original filename components
            original_name = os.path.splitext(file.filename)[0]
            original_ext = os.path.splitext(file.filename)[1] or ".tmp"
            
            # Create a filename that preserves the original name but adds uniqueness
            unique_id = uuid4().hex[:8]
            safe_filename = f"{original_name}_{unique_id}{original_ext}"
            
            # Create full path in temp directory
            temp_path = os.path.join(temp_dir, safe_filename)
            
            # Save the file
            contents = await process_file(file)
            with open(temp_path, "wb") as f:
                f.write(contents)
                
            # Add to result list
            result.append((temp_path, file.filename))
            logger.debug(f"Saved {file.filename} to {temp_path}")
            
        except Exception as e:
            logger.error(f"Failed to save {file.filename}: {e}")
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to process file '{file.filename}': {str(e)}"
            )
    
    return result

@router.post("/analyze", response_model=VisualizerResponse)
async def analyze_files(
    files: List[UploadFile] = File(...),
    prompt: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """
    Endpoint to analyze files (images or PDFs).
    Processes all files and returns a single combined response.
    Can handle multiple files and analyze them together.
    """
    # Ensure agent is initialized
    global visualizer_agent
    if visualizer_agent is None:
        logger.info("Initializing VisualizerAgent on first request")
        visualizer_agent = await init_visualizer_agent()
    
    if not files:
        logger.warning("API request with no files provided")
        raise HTTPException(status_code=400, detail="No files provided")
    
    saved_files = []
    temp_paths = []
    file_details = []  # List to store original filenames for better error reporting
    
    try:
        # Save uploaded files with better naming
        saved_files = await save_uploaded_files(files)
        temp_paths = [path for path, _ in saved_files]
        file_details = [f"{i+1}. {original}" for i, (_, original) in enumerate(saved_files)]
        
        logger.info(f"Analyzing {len(saved_files)} files for user {user_id}")
        logger.debug(f"Files: {', '.join([name for _, name in saved_files])}")
        
        # Create a more descriptive prompt if none provided
        if not prompt:
            files_list = "\n".join(file_details)
            prompt = f"Please analyze these {len(saved_files)} architectural files:\n{files_list}"
        
        # HANDLE MULTIPLE FILES:
        # Use analyze_files method which properly supports multiple files
        if len(temp_paths) > 1:
            logger.info(f"Using multi-file analysis for {len(temp_paths)} files")
            
            # Add a batch size limit for large uploads - process in batches of 2 files if more than 3
            if len(temp_paths) > 3:
                logger.info(f"Large file set detected ({len(temp_paths)} files). Processing in batches.")
                
                # Process first batch of 2 files
                batch_result = await visualizer_agent.analyze_files(
                    file_paths=temp_paths[:2],
                    text=f"Please analyze these architectural files (batch 1 of {(len(temp_paths) + 1) // 2}):\n" + 
                         "\n".join(file_details[:2]),
                    user_id=user_id,
                    session_id=session_id,
                    stream=False
                )
                
                # Process remaining files in a second batch
                remaining_result = await visualizer_agent.analyze_files(
                    file_paths=temp_paths[2:],
                    text=f"Please analyze these architectural files (batch 2 of {(len(temp_paths) + 1) // 2}):\n" + 
                         "\n".join(file_details[2:]),
                    user_id=user_id,
                    session_id=session_id,
                    stream=False
                )
                
                # Combine results with a section separator
                result = f"{batch_result}\n\n## ADDITIONAL FILES ANALYSIS\n\n{remaining_result}\n\n## COMPLETE PROJECT SUMMARY\n\nThis project consists of {len(temp_paths)} files analyzed above."
            else:
                # Standard processing for 2-3 files
                result = await visualizer_agent.analyze_files(
                    file_paths=temp_paths,
                    text=prompt,
                    user_id=user_id,
                    session_id=session_id,
                    stream=False
                )
            
            # Schedule cleanup as a background task
            if background_tasks:
                background_tasks.add_task(cleanup_temp_files, temp_paths)
            else:
                # Fallback for cleanup if background_tasks isn't available
                asyncio.create_task(asyncio.to_thread(cleanup_temp_files, temp_paths))
                
            # Return the analysis result
            return VisualizerResponse(content=result)
            
        # HANDLE SINGLE FILE:
        # Fall back to visualize for single file (with streaming)
        elif temp_paths:
            logger.info(f"Using single-file analysis with streaming for: {saved_files[0][1]}")
            response_generator = await visualizer_agent.visualize(
                text=prompt,
                file_path=temp_paths[0],
                user_id=user_id,
                session_id=session_id
            )
            
            # Combine all responses into a single string
            combined_content = []
            
            # Properly await and collect all responses from the async generator
            async for response in response_generator:
                text = None
                if hasattr(response, "content") and response.content:
                    text = response.content
                elif hasattr(response, "delta") and response.delta:
                    text = response.delta
                if text:
                    combined_content.append(text)
            
            # Join all responses
            final_content = "".join(combined_content)
            
            # Schedule cleanup as a background task
            if background_tasks:
                background_tasks.add_task(cleanup_temp_files, temp_paths)
            else:
                # Fallback for cleanup if background_tasks isn't available
                asyncio.create_task(asyncio.to_thread(cleanup_temp_files, temp_paths))
            
            # Handle empty response
            if not final_content.strip():
                logger.error("Empty analysis result")
                raise HTTPException(status_code=500, detail="Failed to generate analysis")
            
            # Return a single response object
            return VisualizerResponse(content=final_content)
        else:
            raise HTTPException(status_code=400, detail="No valid files processed")
    
    except HTTPException as http_exc:
        # Clean up temp files on error
        if temp_paths:
            asyncio.create_task(asyncio.to_thread(cleanup_temp_files, temp_paths))
        # Re-raise HTTP exceptions
        logger.error(f"HTTP Exception: {http_exc.detail}")
        raise
        
    except Exception as e:
        # Clean up temp files on error
        if temp_paths:
            asyncio.create_task(asyncio.to_thread(cleanup_temp_files, temp_paths))
        # Add better error handling with file details for easier debugging
        error_msg = f"Error processing files ({', '.join([name for _, name in saved_files])}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Error analyzing files: {str(e)}"
        )