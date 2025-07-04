"""
BOQ Agent Module - Generates Bill of Quantities for construction projects.

This module provides a specialized agent for generating detailed Bill of Quantities
based on architectural specifications and project requirements.
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
logger = logging.getLogger("BOQ_AGENT")

# Load environment variables
dotenv.load_dotenv()

# Define constants
INSTRUCTIONS_FILE = 'boq_agent.json'
DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_USER = "default_user"
DEFAULT_SESSION = "default_session"

# def get_instructions() -> Dict[str, Any]:
#     """
#     Load agent instructions from JSON file.
    
#     Returns:
#         Dict[str, Any]: Instructions dictionary
    
#     Raises:
#         FileNotFoundError: If instructions file doesn't exist
#         json.JSONDecodeError: If JSON parsing fails
#     """
#     try:
#         instructions_path = Path(__file__).parent.parent / 'instructions' / INSTRUCTIONS_FILE
#         with open(instructions_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except FileNotFoundError as e:
#         logger.error(f"Instructions file not found: {e}")
#         raise
#     except json.JSONDecodeError as e:
#         logger.error(f"Failed to parse instructions JSON: {e}")
#         raise

# Load BOQ agent instructions
try:
    BOQ_AGENT_INSTRUCTIONS = [
                "You are an expert Quantity Surveyor trained in Time Saver Standards and international BoQ practice. You will be given structured architectural analysis for multiple distinct floor plans (from images or PDFs).",
                
                "CRITICAL: ALWAYS CHECK CHAT HISTORY FOR ARCHITECTURAL DATA!",
                "FIRST ACTION: Use get_chat_history() tool to search for architectural analysis data from previous messages.",
                "SECOND ACTION: Look for detailed room data, dimensions, fixtures, and construction details.",
                "If architectural data found, use it for precise BOQ. If no data found, generate template BOQ.",
                
                "Your task is to produce:",
                "1. A *multi-section BoQ table* for each floor plan, divided by construction discipline.",
                "2. A *final consolidated summary* table, grouped discipline-wise, referencing source plans.",
                "3. A *material-level breakdown table* for each applicable BoQ item (concrete, siding, flooring, plumbing, etc.) to support procurement planning.",

                "--- ENHANCED MEMORY ACCESS FOR DETAILED BOQ ---",
                "Before generating the BoQ, COMPREHENSIVELY access all saved architectural data:",
                
                "=== BASIC PROJECT DATA ===",
                "Check: building_type, total_floors, total_floor_area, project_complexity, architectural_style",
                
                "=== DETAILED ROOM DATA ===", 
                "Check: bedrooms_details, bathrooms_details, kitchens_details, living_areas_details",
                "Check: utility_spaces, special_rooms, room_dimensions, total_rooms_all_floors",
                
                "=== ARCHITECTURAL ELEMENTS ===",
                "Check: doors_breakdown, windows_breakdown, staircases_details, balconies_terraces",
                "Check: columns_beams, wall_details, building_footprint",
                
                "=== MEP SYSTEMS DETAILED ===",
                "Check: plumbing_fixtures_detailed, electrical_detailed, hvac_detailed",
                "Check: plumbing_layout, electrical_layout, mechanical_rooms",
                
                "=== CONSTRUCTION MATERIALS ===",
                "Check: foundation_type, roof_details, exterior_materials, flooring_materials",
                "Check: ceiling_heights, insulation_requirements",
                
                "=== QUANTITY ESTIMATES ===",
                "Check: concrete_estimate, framing_estimate, drywall_estimate, flooring_estimate",
                "Check: roofing_estimate, siding_estimate, paint_estimate",
                
                "=== SPECIALIZED FEATURES ===",
                "Check: fireplaces_details, built_ins, countertops, tile_areas",
                "Check: garage_details, outdoor_features",
                
                "=== MULTI-FILE DATA (if available) ===",
                "Check: total_floor_plans_analyzed, combined_total_area, floors_relationship",
                "Check: total_concrete_needed, total_framing_needed, total_drywall_needed",
                
                "=== BOQ GENERATION STRATEGY ===",
                "- If detailed memory data exists (ready_for_boq=true), generate PRECISE quantities",
                "- Use actual room counts, dimensions, and material estimates from memory",
                "- Reference specific architectural features and MEP requirements",
                "- Include detailed material breakdowns based on actual measurements",
                "- Calculate waste factors based on project complexity from memory",
                "- If limited data available, generate template BOQ and specify what's needed",
                
                "=== MEMORY CONFIRMATION ===",
                "- Always state how many memory items were found and used",
                "- List specific data sources (e.g., 'Based on 3-bedroom, 2-bath analysis')",
                "- Note any missing critical information",
                "- Save BOQ generation metadata:",
                "  - memory.save('boq_generated_from_detailed_analysis', 'true/false')",
                "  - memory.save('boq_accuracy_level', 'high/medium/low based on data available')",
                "  - memory.save('boq_generation_date', 'current date')",

                "--- Main Disciplines ---",
                "Break the BoQ into the following construction categories:",
                "1.0 CIVIL WORKS",
                "2.0 ELECTRICAL WORKS",
                "3.0 PLUMBING & SANITARY",
                "4.0 MECHANICAL / HVAC",
                "5.0 FIXTURES & FURNITURE",
                "6.0 EXTERNAL WORKS / SPECIAL FEATURES",

                "--- Section Examples ---",
                "Each discipline may contain subsections like:",
                "1.0 CIVIL WORKS",
                "  - 1.1 Site Preparation",
                "  - 1.2 Substructure",
                "  - 1.3 Superstructure",
                "  - 1.4 Masonry",
                "  - 1.5 Flooring",
                "  - 1.6 Plaster, Paint & Finishes (includes 1.6.4 Siding)",
                
                "2.0 ELECTRICAL WORKS",
                "  - 2.1 Light Points",
                "  - 2.2 Power Sockets",
                "  - 2.3 Wiring & Conduits",
                "  - 2.4 Switchboards",
                "  - 2.5 Fixtures",

                "3.0 PLUMBING & SANITARY",
                "  - 3.1 Toilets",
                "  - 3.2 Washbasins",
                "  - 3.3 Kitchen & Utility Sinks",
                "  - 3.4 Showers, Bathtubs, Water Heaters",
                "  - 3.5 Water Supply Piping",
                "  - 3.6 Drainage / Venting",

                "4.0 MECHANICAL / HVAC",
                "  - 4.1 Duct Outlets / Grills",
                "  - 4.2 HVAC System",
                "  - 4.3 Thermostats / Dampers",

                "5.0 FIXTURES & FURNITURE",
                "  - 5.1 Kitchen Cabinets",
                "  - 5.2 Bathroom Vanities / Closets",
                "  - 5.3 Kitchen Counters / Islands",

                "6.0 EXTERNAL WORKS",
                "  - 6.1 Decks, Porches",
                "  - 6.2 Paving / Landscaping",
                "  - 6.3 Stone Veneer, Soffit, Roof Fascia",
                "  - 6.4 Exterior Staircases, Rails",

                "--- BoQ Table Format ---",
                "For each plan, generate the following table:",
                "| Item No. | Description                          | Unit  | Quantity | Room/Area (if known) | Notes                     |",
                "|----------|--------------------------------------|-------|----------|----------------------|----------------------------|",

                "--- Detailed Material Breakdown Rules (Global) ---",
                "For every BoQ item that requires physical materials, generate a separate material breakdown table directly below it.",
                "- Use the heading: *Material Breakdown for [Item No. – Description]*",
                "- Format:",
                "| Material | Specification       | Unit | Quantity | Notes |",
                "|----------|---------------------|------|----------|-------|",

                "- Apply waste margins and round up packaging units:",
                "  • Cement: 50 kg bag, Steel: kg, Sand/Aggregate: m³, Water: L",
                "  • Wiring: meters, Conduits: 10 ft or 3m lengths, Nails: per box, Pipes: 3m or 6m",
                "  • Siding Components: use PC, RL, SQ, BX, TB depending on vendor packaging",
                "  • Include notes for rounding, waste, style types",
                
                "- Examples:",
                "*Material Breakdown for 1.2.2 – Reinforced Concrete Footings*",
                "| Material | Specification   | Unit | Quantity | Notes             |",
                "|----------|-----------------|------|----------|-------------------|",
                "| Cement   | OPC 43 Grade    | Bag  | 48       | For PCC & RCC mix |",
                "| Steel    | 12mm Deformed   | kg   | 360      | Footing mesh      |",
                "| Crush    | 1/2\" Aggregate  | m³   | 4.5      | Coarse aggregate  |",
                "| Sand     | Washed River    | m³   | 3.8      | Fine aggregate    |",
                "| Water    | Potable         | L    | 550      | Mixing + curing   |",

                "*Material Breakdown for 1.6.4 – Vinyl Siding (Dutch Lap Style)*",
                "| Material              | Specification   | Unit | Quantity | Notes                     |",
                "|-----------------------|------------------|------|----------|----------------------------|",
                "| Vinyl Siding Panels   | Dutch Lap, White | SQ   | 17.94    | Includes 10% waste         |",
                "| Starter Strip         | 12 ft            | PC   | 18       | For 215.38 LF base         |",
                "| J-Channel             | 12 ft            | PC   | 46       | Around windows & edges     |",
                "| House Wrap            | 9x100 ft roll    | RL   | 2        | Covers full siding area    |",
                "| Flashing Tape         | 75 ft roll       | RL   | 5        | Around openings            |",
                "| Siding Nails          | 2\" Box (50#)     | BX   | 1        | For 20 SQ coverage         |",
                "| Caulk                 | White            | TB   | 2        | 1 tube per 8–10 SQ         |",

                "--- Consolidated Summary Table Format ---",
                "After all floor plans, generate a master summary grouped by item number:",
                "| Item No. | Description                 | Unit | Total Qty | Source Plans        | Notes |",
                "|----------|-----------------------------|------|-----------|---------------------|-------|",

                "--- Additional Rules ---",
                "- Maintain structured item codes: 1.6.4, 2.3.1, etc.",
                "- Prefer SI units unless standard packaging uses imperial (e.g., siding rolls, pipe lengths).",
                "- Round quantities based on packaging standards. Avoid decimal values for pieces or rolls.",
                "- If style or product type (e.g., Shake Siding, Composite Board & Batten) is visible in elevations or annotations, use it.",
                "- If nothing is specified, default siding = Traditional Lap Vinyl.",
                "- Do not calculate or output costs unless explicitly asked.",
                "- Do not include amount and unit price columns in the BoQ tables.",
                "- Send the final BoQ in a markdown code block.",
            ]
    logger.info("Successfully loaded BOQ agent instructions")
except Exception as e:
    logger.error(f"Failed to load instructions: {e}")
    # Provide fallback minimal instructions
    BOQ_AGENT_INSTRUCTIONS = {
        "system_message": "Generate a detailed Bill of Quantities for construction projects."
    }
    logger.warning("Using fallback instructions")

async def initialize_shared_resources():
    """
    Initialize shared memory, storage, and knowledge base asynchronously.
    
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

class BOQAgent(Agent):
    """
    Bill of Quantities Agent specialized in generating detailed construction BOQs.
    
    This agent leverages the Gemini model to generate standardized bill of quantities
    for construction projects based on specifications and architectural details.
    """
    
    def __init__(self, memory=None, storage=None, knowledge=None):
        """
        Initialize the BOQAgent with shared resources.
        
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
        api_key = os.getenv("BOQ_AGENT_API_KEY")
        if not api_key:
            logger.warning("BOQ_AGENT_API_KEY not found in environment")
        
        model_id = os.getenv("BOQ_AGENT_MODEL", DEFAULT_MODEL)
        logger.info(f"Using model: {model_id}")
        
        super().__init__(
            name="BOQAgent",
            agent_id="boq_agent",
            model=Gemini(id=model_id, api_key=api_key),
            memory=memory,
            storage=storage,
            description=(
                "BOQ agent generates detailed Bill of Quantities for construction projects "
                "based on architectural drawings, specifications, and project data. It follows "
                "industry standards for quantity surveying. For now, if not complete details are "
                "provided like MEP or structural details, just provide the BOQ for the architectural "
                "part only."
            ),
            instructions=BOQ_AGENT_INSTRUCTIONS,
            knowledge=knowledge,  # Now using the pre-initialized knowledge
            search_knowledge=True,
            enable_session_summaries=True,
            add_history_to_messages=True,
            add_session_summary_references=True,
            num_history_sessions=5,
            debug_mode=True
        )
        logger.info("BOQAgent initialized successfully")

    async def generate_boq(
        self, 
        data: str, 
        user_id: Optional[str] = None, 
        session_id: Optional[str] = None
    ) -> AsyncIterator[RunResponse]:
        """
        Generate Bill of Quantities based on project data.

        Args:
            data: Project data, specifications, or architectural information
            user_id: User identifier for session management
            session_id: Session identifier for conversation continuity

        Returns:
            AsyncIterator[RunResponse]: Streaming response events with BOQ data
            
        Raises:
            ValueError: If input data is empty
            Exception: For any errors during BOQ generation
        """
        # Validate input parameters
        if not data or not data.strip():
            logger.error("Empty input data provided to generate_boq")
            raise ValueError("Project data cannot be empty")
            
        # Use default values if not provided
        user_id = user_id or DEFAULT_USER
        session_id = session_id or DEFAULT_SESSION
        
        # Truncate long data in logs
        log_data = data[:100] + "..." if len(data) > 100 else data
        logger.info(f"Generating BOQ with data: {log_data}")
        logger.info(f"User ID: {user_id}, Session ID: {session_id}")

        try:
            # Generate BOQ using the agent's run method
            logger.debug("Starting BOQ generation")
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

            logger.info("BOQ generation completed successfully")

        except Exception as e:
            logger.error(f"BOQ generation failed: {e}", exc_info=True)
            raise

async def main():
    """
    Async main function for testing the BOQ agent as a standalone CLI application.
    
    This function provides an interactive CLI interface for testing the BOQ agent
    without requiring the full web API.
    """
    try:
        # Initialize shared resources
        logger.info("Starting BOQ Agent CLI")
        memory, storage, knowledge = await initialize_shared_resources()  # Now unpacking three values
        agent = BOQAgent(memory=memory, storage=storage, knowledge=knowledge)  # Pass knowledge to agent

        print("\n" + "=" * 50)
        print("🏗️  BOQ Agent - Interactive CLI")
        print("=" * 50)
        print("Enter project data for BOQ generation (or type 'exit' to quit)")
        print("-" * 50)

        while True:
            try:
                # Use asyncio-compatible input
                text = await asyncio.to_thread(input, "Enter project data: ")

                if text.lower() in ('exit', 'quit'):
                    print("Thank you for using the BOQ Agent. Goodbye!")
                    break

                if not text.strip():
                    print("Please enter valid project data.")
                    continue

                print(f"\n🔄 Generating BOQ...")
                print("-" * 50)

                # Generate BOQ asynchronously
                events = []
                try:
                    response_generator = agent.generate_boq(
                        text, 
                        user_id="test_user", 
                        session_id="test_session"
                    )
                    
                    async for event in response_generator:
                        events.append(event)
                        
                    if events:
                        pprint_run_response(events, markdown=True)
                    else:
                        print("No response received from the agent.")
                        
                except ValueError as ve:
                    print(f"❌ Input error: {ve}")
                    continue

                print("\n" + "-" * 50)
                print("✅ BOQ generation completed!")
                print()

            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user. Goodbye!")
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