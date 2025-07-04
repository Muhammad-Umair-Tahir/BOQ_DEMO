# agents/visualizer_agent.py

import os
import dotenv
import requests
from typing import Iterator, List, Optional
from agno.agent import Agent, RunResponse
from agno.models.google import Gemini
from agno.media import Image, File
from utility.utils import shared_memory, shared_storage
import asyncio

dotenv.load_dotenv()

shared_memory = asyncio.run(shared_memory())
shared_storage = asyncio.run(shared_storage())

class VisualizerAgent(Agent):
    def __init__(self):
        super().__init__(
            name="VisualizerAgent",
            agent_id="visualizer_agent",
            model=Gemini(
                id=os.getenv("VISUALIZER_AGENT_MODEL", "gemini-1.5-flash"), 
                api_key=os.getenv("VISUALIZER_AGENT_API_KEY")
            ),
            memory=shared_memory,
            storage=shared_storage,
            add_history_to_messages=True,
            enable_user_memories=True,
            add_references=True,
            description="Analyzes visual content from image or PDF URLs.",
            instructions="Fetch and analyze each provided URL (image or PDF) in depth.",
            debug_mode=True
        )

    def visualize(
        self,
        text: Optional[str] = None,
        urls: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Iterator[RunResponse]:
        if not urls:
            yield RunResponse(content="No URLs provided. Please submit image or PDF URLs.")
            return

        pdf_objs, img_objs = [], []

        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                content = resp.content
                if content.startswith(b'%PDF'):
                    pdf_objs.append(File(url=content))
                else:
                    img_objs.append(Image(url=content))
            except Exception as e:
                yield RunResponse(content=f"Failed to fetch or parse URL '{url}': {e}")

        if not pdf_objs and not img_objs:
            yield RunResponse(content="No valid image or PDF found at provided URLs.")
            return

        prompt = text.strip() if text and text.strip() else (
            "Please analyze the provided visual content from the URLs."
        )

        yield from self.run(
            prompt,
            files=pdf_objs or None,
            images=img_objs or None,
            user_id=user_id,
            session_id=session_id,
            stream=True
        )


if __name__ == "__main__":
    agent = VisualizerAgent()
    urls = ["https://ik.imagekit.io/4gwguq3r0/ChatGPT%20Image%20Jun%2027,%202025,%2007_56_29%20PM.png?updatedAt=1751625525577"]
    for response in agent.visualize(text="Analyze these visuals", urls=urls):
        print(response.content)