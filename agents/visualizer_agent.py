import os
import logging
import asyncio
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Optional, Union

import dotenv
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.media import Image, File
from agno.utils.pprint import pprint_run_response

from utility.utils import shared_memory, shared_storage
from utility.rag import rag_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VISUALIZER_AGENT")

dotenv.load_dotenv()

# Default model configurations
DEFAULT_MAIN_MODEL = "gemini-2.0-flash-lite"
DEFAULT_MULTI_FILE_MODEL = "gemini-2.0-flash"
DEFAULT_USER = "default_user"
DEFAULT_SESSION = "default_session"



class VisualizerAgent(Agent):
    def __init__(self, memory=None, storage=None, knowledge=None):
        """
        Initialize the VisualizerAgent with shared resources.
        
        Args:
            memory: Shared memory instance for conversation history
            storage: Shared storage instance for persistent data
            knowledge: Knowledge base for domain-specific information
        """
        # Store the shared resources
        self.memory = memory
        self.storage = storage
        
        # Get API key and model from environment variables with fallbacks
        api_key = os.getenv("VISUALIZER_AGENT_API_KEY", os.getenv("GOOGLE_API_KEY"))
        if not api_key:
            logger.warning("No API key found for VisualizerAgent. Using default credentials.")
            
        main_model_id = os.getenv("VISUALIZER_AGENT_MODEL", DEFAULT_MAIN_MODEL)
        multi_file_model_id = os.getenv("VISUALIZER_MULTI_FILE_MODEL", DEFAULT_MULTI_FILE_MODEL)
        
        
        logger.info(f"Initializing VisualizerAgent with model: {main_model_id}")
        
        # Only initialize the base Agent with minimal info, no instructions
        super().__init__(
            name="VisualizerAgent",
            agent_id="visualizer_agent",
            model=Gemini(
                id=main_model_id,
                api_key=api_key
            ),
            memory=memory,
            storage=storage,
            description="Specialized agent for analyzing architectural drawings, floor plans, and construction documents. Extracts detailed information about rooms, dimensions, fixtures, and building elements to support construction planning.",
            # No instructions here; all logic uses multi_file_agent's instructions
            instructions=[],
            knowledge=knowledge,  # Add RAG knowledge base
            search_knowledge=True,  # Enable knowledge search
            
            # CHAT HISTORY CONFIG - Increased for better conversation continuity
            add_history_to_messages=True,
            num_history_runs=20,  # Remember last 20 conversation turns
            read_chat_history=True,
            
            # MEMORY CONFIG
            enable_agentic_memory=True,
            enable_user_memories=True,
            enable_session_summaries=True,
            
            debug_mode=True,
            expected_output="(Formatted architectural plan summary)"
        )

        logger.info(f"Initializing MultiFileAnalyzer with model: {multi_file_model_id}")
        
        self.multi_file_agent = Agent(
            name="VIAB_MultiFileAnalyzer",
            model=Gemini(
                id=multi_file_model_id,
                api_key=api_key
            ),
            memory=self.memory,  # Use the same memory instance as main agent!
            storage=self.storage,  # Use the same storage instance as main agent!
            knowledge=knowledge,  # Use the same knowledge base
            description="Analyzes multiple architectural files individually and provides consolidated project details for construction planning.",
            
            # CHAT HISTORY CONFIG - Same as main agent for consistency
            add_history_to_messages=True,
            num_history_runs=20,  # Remember last 20 conversation turns
            read_chat_history=True,
            
            # MEMORY CONFIG - Same as main agent
            enable_agentic_memory=True,
            enable_user_memories=True,
            enable_session_summaries=True,
            search_knowledge=True,  # Enable knowledge search
            
            debug_mode=True,
            
            instructions=[    
                "You are an expert architectural analyst specializing in interpreting multiple floor plan images and architectural documents. Your primary responsibility is to analyze each distinct floor plan individually, whether they come from separate files or multiple layouts within a single file (such as multi-page PDFs).",
                "Your analysis focuses on providing detailed architectural information that supports construction planning and project development.",
                
                # CRITICAL MEMORY MANAGEMENT FOR MULTI-FILE ANALYSIS
                "MEMORY MANAGEMENT - SAVE COMPREHENSIVE CONSOLIDATED DATA:",
                "After analyzing all files, save DETAILED consolidated information to memory:",
                "Use the same detailed approach as single file analysis but consolidate across all plans:",
                
                "=== CONSOLIDATED BUILDING DATA ===",
                "- memory.save('total_floor_plans_analyzed', 'exact number with file names')",
                "- memory.save('building_type_consolidated', 'overall project type with variations')",
                "- memory.save('combined_total_area', 'sum of all floor areas with breakdown by floor')",
                "- memory.save('floors_relationship', 'how floors relate: basement, ground, upper')",
                
                "=== CONSOLIDATED ROOM INVENTORY ===",
                "- memory.save('total_rooms_all_floors', 'comprehensive room count across all plans')",
                "- memory.save('bedrooms_all_floors_detailed', 'all bedrooms with sizes and floor locations')",
                "- memory.save('bathrooms_all_floors_detailed', 'all bathrooms with types and floor locations')",
                "- memory.save('kitchens_all_floors_detailed', 'all kitchens with layouts and floor locations')",
                "- memory.save('common_areas_detailed', 'living rooms, dining rooms across all floors')",
                
                "=== CONSOLIDATED ARCHITECTURAL ELEMENTS ===",
                "- memory.save('total_doors_detailed', 'all doors by type and floor with sizes')",
                "- memory.save('total_windows_detailed', 'all windows by type and floor with sizes')",
                "- memory.save('staircases_all_floors', 'between-floor connections, types, materials')",
                "- memory.save('structural_system', 'overall structural approach across floors')",
                
                "=== CONSOLIDATED MEP SYSTEMS ===",
                "- memory.save('plumbing_system_detailed', 'complete plumbing fixture inventory by floor')",
                "- memory.save('electrical_system_detailed', 'complete electrical load and distribution')",
                "- memory.save('hvac_system_detailed', 'complete HVAC requirements and distribution')",
                "- memory.save('mechanical_rooms', 'utility spaces and equipment locations')",
                
                "=== CONSOLIDATED CONSTRUCTION QUANTITIES ===",
                "- memory.save('total_concrete_needed', 'foundation + slabs for all floors')",
                "- memory.save('total_framing_needed', 'lumber for entire structure')",
                "- memory.save('total_drywall_needed', 'all interior walls and ceilings')",
                "- memory.save('total_flooring_needed', 'by material type across all floors')",
                "- memory.save('total_roofing_needed', 'complete roof system requirements')",
                "- memory.save('total_exterior_cladding', 'siding, brick, stone for entire building')",
                
                "=== PROJECT COMPLEXITY ASSESSMENT ===",
                "- memory.save('multi_file_analysis_complete', 'true')",
                "- memory.save('project_complexity_detailed', 'complexity rating with specific reasons')",
                "- memory.save('construction_challenges', 'identified difficult aspects')",
                "- memory.save('design_consistency', 'how well plans work together')",
                
                "ALWAYS confirm comprehensive data storage with count of items saved.",
                
                "--- File and Layout Detection ---",
                "1. **Detect and Interpret All Uploaded Files**:",
                "   - Identify file type: single floor plan image, multi-page PDF, or architectural document.",
                "   - For PDFs, check each page and layout for unique floor plans.",
                "   - Detect layouts embedded on a single page (e.g., Layout A and Layout B).",
                
                "2. **Treat Each Unique Floor Plan Separately**:",
                "   - Label each plan clearly (e.g., Floor Plan 1: Page 1 of abc.pdf).",
                "   - Analyze each as a standalone plan, regardless of shared files.",
                
                "--- Detailed Per-Plan Architectural Analysis ---",
                "3. **Extract the Following From Each Floor Plan**:",
                
                "**A. Layout Classification**",
                "- Project Type: Residential, Commercial, Mixed-Use, etc.",
                "- Structure Type: Single-story, Multi-story, Duplex, Option A/B",
                
                "**B. Room and Zone Identification**",
                "- List all labeled or visually identifiable spaces: bedrooms, kitchens, lobbies, balconies, toilets, stairs, terraces, etc.",
                
                "**C. Dimensions and Area Estimation**",
                "- If dimensions are written, extract and calculate total and per-room area.",
                "- **If dimensions are NOT written**, estimate sizes **according to Time Saver Standards**. Examples:",
                "  - Standard Bedroom: ~3.0m x 3.6m (10x12 ft)",
                "  - Small Bathroom: ~1.5m x 2.1m (5x7 ft)",
                "  - Living Room: ~3.5m x 5.0m (12x16 ft)",
                "- Always clearly mark such cases as '**Standard Estimate – No Dimension Provided**' in the notes.",
                
                "**D. Architectural Elements Count**",
                "- Interior Doors",
                "- Exterior Doors",
                "- Windows (sliding, fixed, casement – if known)",
                "- Stairs, Balconies, Columns, Load-bearing walls, Shafts",
                
                "**E. Built-in Fixtures and Furniture**",
                "- Washbasins, Toilets, Bathtubs, Kitchen Counters, Closets, Cabinets, Storage",
                
                "**F. Electrical Symbols (if visible)**",
                "- Light Points, Sockets, Switchboards, Distribution Boards, Ceiling Fans",
                
                "**G. Plumbing and Drainage Features**",
                "- Water Outlets, Toilets, Sinks, Showers, Geysers, Utility Zones",
                
                "**H. HVAC and Mechanical Features**",
                "- A/C Units, Ducts, Mechanical Rooms, Grills, Lifts, Exhaust Systems",
                
                "--- Output Format ---",
                "4. **Markdown Structure Per Plan**:",
                
                "### File-by-File Analysis",
                
                "#### 📄 FLOOR PLAN 1: [Page 1 of filename.pdf] OR [filename1.jpg]",
                "**Source**: [filename + page]",
                "**Floor Plan Type**: Residential / Commercial / Mixed",
                "**Layout Description**: Single floor / Option A / Ground Floor, etc.",
                
                "**Rooms & Spaces**:",
                "- Bedroom 1, Kitchen, Toilet, Lobby, Balcony, etc.",
                
                "**Dimensions & Area**:",
                "- Bedroom 1: 12' x 10' = 120 sq ft",
                "- Bathroom: Estimated 5' x 7' = 35 sq ft (**Standard Estimate – No Dimension Provided**)",
                "- Total Estimated Area: 820–860 sq ft",
                
                "**Architectural Elements**:",
                "- Windows: 5 (2 Sliding, 3 Fixed)",
                "- Interior Doors: 4",
                "- Exterior Doors: 2",
                "- Stairs: 1 (DN labeled)",
                
                "**Plumbing Fixtures**:",
                "- Toilets: 2",
                "- Washbasins: 2",
                "- Kitchen Sink: 1",
                
                "**Electrical Fixtures**:",
                "- Light Points: 6 (assumed 1 per room)",
                "- Switchboards: 3",
                
                "**Special Features**:",
                "- Built-in Wardrobe in Bedroom 1",
                "- L-shaped Kitchen Counter",
                "- HVAC shaft visible in top-right",
                
                "[Repeat for each floor plan]",
                
                "### 🏗️ CONSOLIDATED PROJECT SUMMARY",
                "- **Total Floor Plans Analyzed**: [3]",
                "- **Combined Area**: [~2,600 sq ft, including estimates]",
                "- **Total Rooms**: [18 rooms, 5 toilets, 3 kitchens]",
                "- **Total Architectural Elements**: [15 doors, 20 windows, 3 balconies]",
                "- **Estimation Notes**: Dimensions missing in 2 plans – estimated based on Time Saver norms",
                "- **Plan Relationships**: Floor Plan 1 and 2 are stacked floors; Floor Plan 3 is an alternate layout",
                
                "--- Critical Notes ---",
                "- **If dimensions are missing**, estimate reasonably from known objects (e.g., door widths, wall thickness) or use standard room sizes.",
                "- Clearly flag all assumptions or interpolations.",
                "- Maintain clean formatting for BoQ processing later."],
            markdown=True
        )



    async def analyze_files(self, file_paths: Optional[List[str]] = None, text: Optional[str] = None, 
                     user_id: Optional[str] = None, session_id: Optional[str] = None, 
                     stream: bool = False) -> Union[str, AsyncIterator[RunResponse]]:
        """
        Unified file analysis method that handles single or multiple files seamlessly.
        
        Args:
            file_paths: List of file paths to analyze (can be single file or multiple)
            text: Optional text prompt for analysis
            user_id: User identifier for session management
            session_id: Session identifier for conversation continuity
            stream: Whether to return streaming response or complete response
            
        Returns:
            Analysis result as string or Iterator[RunResponseEvent] if streaming
        """
        print(f"🔍 DEBUG: VisualizerAgent.analyze_files() called")
        print(f"🔍 DEBUG: User ID: {user_id}")
        print(f"🔍 DEBUG: Session ID: {session_id}")
        print(f"🔍 DEBUG: Agent Memory: {self.memory}")
        print(f"🔍 DEBUG: Multi-agent Memory: {self.multi_file_agent.memory}")
        print(f"🔍 DEBUG: Memory objects are same? {self.memory is self.multi_file_agent.memory}")
        
        # Handle the case where no files are provided
        if not file_paths:
            if text:
                if stream:
                    return await self.arun(text, user_id=user_id, session_id=session_id, stream=True)
                else:
                    response = await self.arun(text, user_id=user_id, session_id=session_id, stream=False)
                    return response.content if hasattr(response, 'content') else str(response)
            else:
                raise ValueError("Either file_paths or text must be provided.")
        
        # Ensure file_paths is a list (handle single file case)
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        print(f"🔍 DEBUG: Processing {len(file_paths)} file(s)")
        
        try:
            images = []
            files = []
            file_descriptions = []
            valid_files = 0

            for i, file_path in enumerate(file_paths):
                if not os.path.exists(file_path):
                    print(f"⚠️ WARNING: File not found: {file_path}")
                    continue
                    
                file_ext = os.path.splitext(file_path)[1].lower()
                file_name = os.path.basename(file_path)

                if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                    try:
                        images.append(Image(filepath=file_path))
                        file_descriptions.append(f"Floor Plan {valid_files+1}: {file_name} (Image)")
                        print(f"✅ Added image: {file_name}")
                        valid_files += 1
                    except Exception as e:
                        print(f"❌ ERROR: Failed to process image {file_name}: {e}")
                        file_descriptions.append(f"{file_name} (Image - Error)")

                elif file_ext == '.pdf':
                    try:
                        files.append(File(filepath=file_path))
                        file_descriptions.append(f"Floor Plan {valid_files+1}: {file_name} (PDF Document)")
                        print(f"✅ Added PDF: {file_name}")
                        valid_files += 1
                    except Exception as e:
                        print(f"❌ ERROR: Failed to process PDF {file_name}: {e}")
                        file_descriptions.append(f"{file_name} (PDF - Error)")

                else:
                    print(f"⚠️ WARNING: Unsupported file type: {file_ext}")
                    file_descriptions.append(f"{file_name} (Unsupported)")

            if valid_files == 0:
                raise ValueError("No valid files found for analysis")

            # Create appropriate prompt based on number of files
            if valid_files == 1:
                file_list = file_descriptions[0] if file_descriptions else "Single file"
                analysis_type = "single file"
                prompt = f"""Please analyze this architectural file in detail.

FILE TO ANALYZE:
- {file_list}

ANALYSIS REQUIREMENTS:
1. Provide comprehensive architectural analysis
2. Identify rooms, elements, dimensions, fixtures
3. Extract all visible details and measurements
4. Follow structured markdown format
5. Include recommendations for construction planning

{f"Additional context: {text}" if text else ""}
"""
            else:
                file_list = "\n".join(f"- {desc}" for desc in file_descriptions if not desc.endswith("(Unsupported)"))
                analysis_type = "multiple files"
                prompt = f"""Please analyze these {valid_files} architectural files individually and then provide a consolidated summary.

FILES TO ANALYZE:
{file_list}

ANALYSIS REQUIREMENTS:
1. Analyze each file separately
2. Use the exact filename in headers
3. Identify rooms, elements, dimensions, fixtures
4. Follow structured markdown format
5. Provide a consolidated project summary

{f"Additional context: {text}" if text else ""}
"""

            print(f"🔍 Running {analysis_type} analysis...")

            # Use the multi_file_agent for comprehensive analysis
            if stream:
                response = await self.multi_file_agent.arun(
                    prompt,
                    images=images if images else None,
                    files=files if files else None,
                    user_id=user_id,
                    session_id=session_id,
                    stream=True
                )
                return response
            else:
                response = await self.multi_file_agent.arun(
                    prompt,
                    images=images if images else None,
                    files=files if files else None,
                    user_id=user_id,
                    session_id=session_id,
                    stream=False
                )
                
                result = response.content if hasattr(response, 'content') else str(response)
                print(f"✅ Analysis complete for {valid_files} file(s)")
                return result

        except Exception as e:
            print(f"❌ ERROR: Exception in analyze_files(): {e}")
            raise

    # Backward compatibility methods
    async def analyze_multiple_files(self, file_paths: List[str]) -> str:
        """Backward compatibility method - delegates to analyze_files"""
        return await self.analyze_files(file_paths=file_paths, stream=False)

    async def visualize(
        self, 
        text: Optional[str] = None, 
        file_path: Optional[str] = None,
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None
    ) -> AsyncIterator[RunResponse]:
        """
        Backward compatibility method for streaming visualization.
        Now delegates to the unified analyze_files method.
        """
        print(f"[DEBUG]: Visualizing - text: {text}, file_path: {file_path}")
        
        if file_path:
            # Convert single file path to list and use streaming
            return await self.analyze_files(
                file_paths=[file_path], 
                text=text, 
                user_id=user_id, 
                session_id=session_id, 
                stream=True
            )
        elif text:
            return await self.analyze_files(
                text=text, 
                user_id=user_id, 
                session_id=session_id, 
                stream=True
            )
        else:
            raise ValueError("Either text or file_path must be provided.")


# Function to initialize shared resources asynchronously
async def initialize_shared_resources():
    """
    Initializes shared memory, storage and knowledge base asynchronously.
    
    Returns:
        Tuple: (memory, storage, knowledge) instances for agent initialization
        
    Raises:
        Exception: If resource initialization fails
    """
    try:
        logger.info("Initializing shared memory, storage and knowledge base")
        memory = await shared_memory()
        storage = await shared_storage()
        knowledge = await rag_agent()  # Now properly awaiting the async rag_agent
        logger.info("Shared resources initialized successfully")
        return memory, storage, knowledge
    except Exception as e:
        logger.error(f"Failed to initialize shared resources: {e}")
        raise