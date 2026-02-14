import aiohttp
import asyncio
import time
from typing import Dict, List, Any, Optional
from loguru import logger


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(
        self, base_url: str = "https://ollama.com/api", api_key: Optional[str] = None
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        await self._ensure_session()

        url = f"{self.base_url}/api/generate"
        data = {"model": model, "prompt": prompt, **kwargs}

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def chat(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """Chat with Ollama model"""
        await self._ensure_session()

        url = f"{self.base_url}/api/chat"
        data = {"model": model, "messages": messages, **kwargs}

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

    async def list_models(self) -> List[str]:
        """List available models"""
        await self._ensure_session()

        url = f"{self.base_url}/api/tags"

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                return [model["name"] for model in result.get("models", [])]
        except Exception as e:
            logger.error(f"Ollama list models error: {e}")
            return []

    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama"""
        await self._ensure_session()

        url = f"{self.base_url}/api/pull"
        data = {"name": model}

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.session.post(url, json=data, headers=headers) as response:
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Ollama pull model error: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test connection to Ollama"""
        await self._ensure_session()

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self.session.get(
                f"{self.base_url}/api/tags", headers=headers
            ) as response:
                return response.status == 200
        except:
            return False


class OllamaAgent:
    """Agent that uses Ollama models for reasoning and response generation"""

    def __init__(
        self,
        agent_type: str,
        model_name: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        import os

        self.agent_type = agent_type
        # Use environment variables for model configuration
        self.model_name = model_name or os.getenv("MODEL_NAME", "glm-4.6:cloud")
        base_url = ollama_base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        # Get API key for cloud-based services (e.g., Ollama Cloud API)
        ollama_api_key = api_key or os.getenv("OLLAMA_API_KEY", "")
        self.ollama_client = OllamaClient(
            base_url, api_key=ollama_api_key if ollama_api_key else None
        )

    async def initialize(self) -> bool:
        """Initialize the agent and check model availability"""
        # Test connection
        if not await self.ollama_client.test_connection():
            logger.error(f"Cannot connect to Ollama at {self.ollama_client.base_url}")
            return False

        # Check if model is available
        models = await self.ollama_client.list_models()
        if self.model_name not in models:
            logger.warning(f"Model {self.model_name} not found, attempting to pull...")
            if not await self.ollama_client.pull_model(self.model_name):
                logger.error(f"Failed to pull model {self.model_name}")
                return False

        logger.info(f"Ollama agent initialized with model {self.model_name}")
        return True

    async def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate a response using the Ollama model"""
        full_prompt = f"Agent Type: {self.agent_type}\n"
        if context:
            full_prompt += f"Context: {context}\n"
        full_prompt += f"Task: {prompt}\n\nPlease provide a helpful response:"

        # Track LLM call with AgentOps if available
        start_time = time.time()
        try:
            result = await self.ollama_client.generate(
                model=self.model_name, prompt=full_prompt, stream=False
            )

            duration = time.time() - start_time
            response = result.get("response", "")

            # Track LLM call
            try:
                import agentops

                agentops.track(
                    "LLM Call",
                    {
                        "agent_type": self.agent_type,
                        "model": self.model_name,
                        "prompt_length": len(full_prompt),
                        "response_length": len(response),
                        "duration": duration,
                    },
                )
            except (ImportError, AttributeError):
                pass

            # Record metrics if metrics collector is available
            if hasattr(self, "_metrics_collector") and self._metrics_collector:
                self._metrics_collector.record_llm_call(
                    agent_id=getattr(self, "agent_id", "unknown"),
                    agent_name=getattr(self, "agent_name", "OllamaAgent"),
                    agent_type=self.agent_type,
                    model=self.model_name,
                    prompt=full_prompt,
                    response=response,
                    duration=duration,
                    success=True,
                )

            print("response from ollama:", response)
            return response
        except Exception as e:
            # Track LLM error
            try:
                import agentops

                duration = time.time() - start_time
                agentops.track(
                    "LLM Error",
                    {
                        "agent_type": self.agent_type,
                        "model": self.model_name,
                        "error": str(e),
                        "duration": duration,
                    },
                )
            except (ImportError, AttributeError):
                pass

            # Record error metrics if available
            if hasattr(self, "_metrics_collector") and self._metrics_collector:
                self._metrics_collector.record_llm_call(
                    agent_id=getattr(self, "agent_id", "unknown"),
                    agent_name=getattr(self, "agent_name", "OllamaAgent"),
                    agent_type=self.agent_type,
                    model=self.model_name,
                    prompt=full_prompt,
                    response="",
                    duration=duration,
                    success=False,
                    error=str(e),
                )

            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

    async def analyze_task(self, task_description: str) -> Dict[str, Any]:
        """Analyze a task and determine approach"""
        prompt = f"""
        Analyze the following task and provide a structured response:
        
        Task: {task_description}
        
        Please provide:
        1. A brief summary of what needs to be done
        2. The main steps required
        3. Any tools or resources needed
        4. Estimated complexity (simple, moderate, complex)
        
        Format your response as JSON-like structure with keys: summary, steps, tools_needed, complexity
        """

        response = await self.generate_response(prompt)

        # Simple parsing (in production, use proper JSON parsing)
        try:
            # Extract structured information from the response
            lines = response.split("\n")
            analysis = {
                "summary": "",
                "steps": [],
                "tools_needed": [],
                "complexity": "moderate",
            }

            current_section = None
            for line in lines:
                line = line.strip()
                if "summary" in line.lower():
                    current_section = "summary"
                elif "steps" in line.lower():
                    current_section = "steps"
                elif "tools" in line.lower():
                    current_section = "tools_needed"
                elif "complexity" in line.lower():
                    current_section = "complexity"
                elif line and current_section:
                    if current_section == "summary":
                        analysis["summary"] = line
                    elif current_section == "steps" and line.startswith("-"):
                        analysis["steps"].append(line[1:].strip())
                    elif current_section == "tools_needed" and line.startswith("-"):
                        analysis["tools_needed"].append(line[1:].strip())
                    elif current_section == "complexity":
                        analysis["complexity"] = line

            return analysis

        except Exception as e:
            logger.error(f"Error parsing task analysis: {e}")
            return {
                "summary": task_description,
                "steps": ["Execute the task as described"],
                "tools_needed": [],
                "complexity": "moderate",
            }

    async def close(self):
        """Close the Ollama client"""
        await self.ollama_client.close()
