import asyncio
import requests
import math
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from loguru import logger


class BaseTool(ABC):
    """Base class for all agent tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        pass


class WebScraperTool(BaseTool):
    """Tool for scraping web content"""

    def __init__(self):
        super().__init__("web_scraper", "Scrape content from web URLs")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

    async def execute(
        self,
        url: str,
        selector: Optional[str] = None,
        max_length: int = 10000,
        **kwargs,
    ) -> Dict[str, Any]:
        """Scrape content from a web URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = soup.find("title")
            title_text = title.get_text().strip() if title else "No title found"

            # Extract content
            if selector:
                elements = soup.select(selector)
                content = "\n".join([elem.get_text().strip() for elem in elements])
            else:
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                content = soup.get_text()
                # Clean up whitespace
                content = "\n".join(
                    line.strip() for line in content.splitlines() if line.strip()
                )

            # Truncate if too long
            if len(content) > max_length:
                content = content[:max_length] + "..."

            return {
                "success": True,
                "url": url,
                "title": title_text,
                "content": content,
                "content_length": len(content),
            }

        except Exception as e:
            logger.error(f"Web scraping error for {url}: {e}")
            return {"success": False, "error": str(e), "url": url}


class SummarizationTool(BaseTool):
    """Tool for summarizing text content"""

    def __init__(self):
        super().__init__("summarize", "Summarize text content")

    async def execute(
        self, text: str, max_sentences: int = 3, strategy: str = "extract", **kwargs
    ) -> Dict[str, Any]:
        """Summarize text content"""
        try:
            sentences = self._split_sentences(text)

            if len(sentences) <= max_sentences:
                summary = text
            else:
                if strategy == "extract":
                    # Simple extractive summarization - take first and last sentences
                    middle_count = max_sentences - 2
                    if middle_count <= 0:
                        summary = sentences[0]
                    else:
                        middle_indices = len(sentences) // middle_count
                        middle_sentences = [
                            sentences[i]
                            for i in range(
                                middle_indices - 1,
                                len(sentences) - 1,
                                len(sentences) // middle_count,
                            )
                        ][:middle_count]
                        summary = (
                            sentences[0]
                            + " "
                            + " ".join(middle_sentences)
                            + " "
                            + sentences[-1]
                        )
                else:
                    # Take first N sentences
                    summary = " ".join(sentences[:max_sentences])

            return {
                "success": True,
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text) if len(text) > 0 else 0,
            }

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"success": False, "error": str(e)}

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]


class MathTool(BaseTool):
    """Tool for mathematical calculations"""

    def __init__(self):
        super().__init__("math", "Perform mathematical calculations")

    async def execute(self, expression: str, **kwargs) -> Dict[str, Any]:
        """Evaluate mathematical expression"""
        try:
            # Safe evaluation - only allow certain operations
            allowed_functions = {
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "exp": math.exp,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
            }

            # Create safe namespace
            safe_dict = {"__builtins__": {}, "math": math, **allowed_functions}

            # Evaluate expression
            result = eval(expression, safe_dict)

            return {"success": True, "result": result, "expression": expression}

        except Exception as e:
            logger.error(f"Math calculation error: {e}")
            return {"success": False, "error": str(e), "expression": expression}


class SearchTool(BaseTool):
    """Tool for searching information"""

    def __init__(self):
        super().__init__("search", "Search for information")

    async def execute(
        self, query: str, max_results: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """Search for information (mock implementation)"""
        try:
            # Mock search results - in real implementation, integrate with search API
            mock_results = [
                {
                    "title": f"Result for: {query}",
                    "snippet": f"This is a mock search result for the query '{query}'. In a real implementation, this would connect to a search engine.",
                    "url": f"https://example.com/search?q={query}",
                }
            ]

            results = mock_results * min(max_results, len(mock_results))

            return {
                "success": True,
                "query": query,
                "results": results[:max_results],
                "total_results": len(results),
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"success": False, "error": str(e), "query": query}


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools"""
        self.register(WebScraperTool())
        self.register(SummarizationTool())
        self.register(MathTool())
        self.register(SearchTool())

    def register(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Tool '{tool.name}' registered: {tool.description}")

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        return await tool.execute(**kwargs)

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools"""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self.tools.values()
        ]

    def get_tool(self, tool_name: str) -> BaseTool:
        """Get a specific tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        return self.tools[tool_name]
