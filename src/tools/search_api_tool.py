import aiohttp
import asyncio
import re
import time
from typing import Dict, List, Any, Optional
from urllib.parse import quote
from bs4 import BeautifulSoup
from loguru import logger

# Import observability instrumentation
try:
    from ..observability import (
        create_tool_span,
        is_agentops_initialized
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


class SearchApiTool:
    """Alternative search using multiple search APIs"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
            }
            self.session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            )

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _duckduckgo_search(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (less restrictive than Google)"""
        await self._ensure_session()

        # DuckDuckGo search URL
        encoded_query = quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(
                        f"DuckDuckGo search failed with status: {response.status}"
                    )
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []

                # DuckDuckGo result structure
                search_results = soup.find_all("div", class_="result")

                for result in search_results[:num_results]:
                    try:
                        # Extract title and link
                        title_tag = result.find("a", class_="result__a")
                        if not title_tag:
                            continue

                        title = title_tag.get_text().strip()
                        url = title_tag.get("href", "")

                        # Extract snippet
                        snippet_tag = result.find("a", class_="result__snippet")
                        if snippet_tag:
                            snippet = snippet_tag.get_text().strip()
                        else:
                            # Alternative snippet extraction
                            snippet_div = result.find("div", class_="result__snippet")
                            snippet = (
                                snippet_div.get_text().strip() if snippet_div else ""
                            )

                        # Clean up snippet
                        snippet = re.sub(r"\s+", " ", snippet)

                        if title and snippet and not url.startswith("#"):
                            results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                    "source": "DuckDuckGo",
                                }
                            )

                    except Exception as e:
                        logger.debug(f"Error parsing DuckDuckGo result: {e}")
                        continue

                return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _brave_search(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search using Brave Search API"""
        await self._ensure_session()

        # Brave search URL
        encoded_query = quote(query)
        url = f"https://search.brave.com/search?q={encoded_query}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Brave search failed with status: {response.status}")
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []

                # Brave result structure
                search_results = soup.find_all(
                    "div", {"id": lambda x: x and x.startswith("result-")}
                )

                for result in search_results[:num_results]:
                    try:
                        # Extract title and link
                        title_tag = result.find("a")
                        if not title_tag:
                            continue

                        title = title_tag.get_text().strip()
                        url = title_tag.get("href", "")

                        # Extract snippet
                        snippet_div = result.find("div", class_="snippet-content")
                        snippet = snippet_div.get_text().strip() if snippet_div else ""

                        # Clean up snippet
                        snippet = re.sub(r"\s+", " ", snippet)

                        if title and snippet and not url.startswith("#"):
                            results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                    "source": "Brave",
                                }
                            )

                    except Exception as e:
                        logger.debug(f"Error parsing Brave result: {e}")
                        continue

                return results

        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []

    async def _wikipedia_search(
        self, query: str, max_results: int = 1
    ) -> List[Dict[str, Any]]:
        """Search using Wikipedia API"""
        import json

        await self._ensure_session()

        # Wikipedia API URL
        encoded_query = quote(query)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []

                data = await response.json()

                if data:
                    return [
                        {
                            "title": data.get("title", ""),
                            "url": data.get("content_urls", {})
                            .get("desktop", {})
                            .get("page", ""),
                            "snippet": data.get("extract", ""),
                            "source": "Wikipedia",
                        }
                    ]

                return []

        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []

    async def search_multiple_sources(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Search across multiple sources and combine results with AgentOps instrumentation."""
        logger.info(f"Searching across multiple sources for: {query}")

        # Use tool span for RAG/search operations
        if OBSERVABILITY_AVAILABLE:
            async with create_tool_span(
                tool_name="search_multiple_sources",
                agent_name="SearchApiTool",
                parameters={"query": query, "max_results": max_results}
            ) as tool_span:
                result = await self._do_search_multiple_sources(query, max_results)
                tool_span.result = f"Found {result.get('total_results', 0)} results"
                return result
        else:
            return await self._do_search_multiple_sources(query, max_results)

    async def _do_search_multiple_sources(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Internal implementation of multi-source search."""
        all_results = []

        # Try different search engines
        search_engines = [
            ("DuckDuckGo", self._duckduckgo_search),
            ("Brave", self._brave_search),
            ("Wikipedia", self._wikipedia_search),
        ]

        for engine_name, search_func in search_engines:
            try:
                logger.info(f"Trying {engine_name} search...")
                results = await search_func(
                    query, max_results // len(search_engines) + 1
                )

                if results:
                    all_results.extend(results)
                    logger.info(f"{engine_name} returned {len(results)} results")

                    # If we got results, we might not need the other engines
                    if len(all_results) >= max_results:
                        break
                else:
                    logger.info(f"{engine_name} returned no results")

                # Brief delay between searches
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"{engine_name} search failed: {e}")

        # Remove duplicates (by URL)
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        # Limit to max results
        unique_results = unique_results[:max_results]

        return {
            "success": len(unique_results) > 0,
            "query": query,
            "results": unique_results,
            "total_results": len(unique_results),
        }


class ResearchReportGenerator:
    """Generates comprehensive research reports from search results"""

    def __init__(self):
        self.search_tool = SearchApiTool()

    async def generate_report(self, topic: str, max_sources: int = 5) -> Dict[str, Any]:
        """Generate a comprehensive research report with AgentOps instrumentation."""
        logger.info(f"Generating research report for topic: {topic}")

        # Use tool span for report generation
        if OBSERVABILITY_AVAILABLE:
            async with create_tool_span(
                tool_name="generate_research_report",
                agent_name="ResearchReportGenerator",
                parameters={"topic": topic, "max_sources": max_sources}
            ) as tool_span:
                result = await self._do_generate_report(topic, max_sources)
                tool_span.result = f"Report generated with {result.get('confidence', 0)} confidence"
                return result
        else:
            return await self._do_generate_report(topic, max_sources)

    async def _do_generate_report(self, topic: str, max_sources: int = 5) -> Dict[str, Any]:
        """Internal implementation of report generation."""
        # Perform multi-source search
        search_data = await self.search_tool.search_multiple_sources(topic, max_sources)

        if not search_data["success"]:
            return {
                "topic": topic,
                "summary": f"Unable to generate research report: {search_data.get('error', 'No search results found')}",
                "sources": [],
                "key_points": [],
                "confidence": 0.0,
                "error": search_data.get("error"),
            }

        # Synthesize report
        report = self._synthesize_report(topic, search_data["results"])

        # Add sources information
        search_engines_used = set(
            r.get("source", "Unknown") for r in search_data["results"]
        )
        report["search_engines"] = list(search_engines_used)

        return report

    def _synthesize_report(
        self, topic: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize a report from search results"""
        if not results:
            return {
                "topic": topic,
                "summary": "No relevant information found for this topic.",
                "sources": [],
                "key_points": [],
                "confidence": 0.0,
            }

        # Collect text from all sources
        all_text = []
        source_data = []

        for result in results:
            text = result.get("snippet", "")
            all_text.append(text)

            source_data.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": text,
                    "type": "web",
                    "source": result.get("source", "Web"),
                }
            )

        # Generate summary using the collected text
        combined_text = ". ".join(all_text)
        summary = self._create_better_summary(combined_text, topic)

        # Extract key points
        key_points = self._extract_key_points(all_text, topic)

        # Calculate confidence with more factors
        confidence = self._calculate_confidence(results)

        return {
            "topic": topic,
            "summary": summary,
            "sources": source_data,
            "key_points": key_points,
            "confidence": confidence,
        }

    def _create_better_summary(self, text: str, topic: str) -> str:
        """Create a better summary for the topic"""
        # Split into sentences
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        # Filter relevant sentences
        topic_words = set(topic.lower().split())
        relevant_sentences = []

        for sentence in sentences:
            if len(sentence) > 20:  # Skip very short sentences
                sentence_lower = sentence.lower()
                words_in_sentence = set(sentence_lower.split())

                # Count topic word matches
                matches = len(topic_words.intersection(words_in_sentence))

                # Add important keywords that suggest significance
                importance_words = [
                    "important",
                    "significant",
                    "key",
                    "major",
                    "recent",
                    "latest",
                    "new",
                    "breakthrough",
                ]
                importance_score = sum(
                    1 for word in importance_words if word in sentence_lower
                )

                # Score the sentence
                score = matches + importance_score

                if score > 0:
                    relevant_sentences.append((sentence, score))

        # Sort by score and take top 3
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in relevant_sentences[:3]]

        if top_sentences:
            return ". ".join(top_sentences) + "."
        else:
            # Fallback to first few sentences if no scoring worked
            return ". ".join(sentences[:3]) + "."

    def _extract_key_points(self, text_list: List[str], topic: str) -> List[str]:
        """Extract key points from the text"""
        key_points = []

        # Combine all text
        combined_text = ". ".join(text_list)
        sentences = [
            s.strip() for s in combined_text.split(".") if 40 < len(s.strip()) < 150
        ]

        # Look for sentences with important indicators
        importance_patterns = [
            r"\b(important|significant|major|key|critical|essential)\b",
            r"\b(found|discovered|developed|created|released|launched)\b",
            r"\b(breakthrough|advancement|innovation|milestone)\b",
        ]

        for sentence in sentences:
            for pattern in importance_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    if sentence not in key_points:
                        key_points.append(sentence)
                        break

        # If no key points found with patterns, extract topic-related sentences
        if not key_points:
            topic_lower = topic.lower()
            for sentence in sentences:
                if any(word in sentence.lower() for word in topic_lower.split()[:2]):
                    if len(key_points) < 3 and sentence not in key_points:
                        key_points.append(sentence)

        return key_points[:5]  # Return up to 5 key points

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on sources quality"""
        if not results:
            return 0.0

        base_confidence = 0.3

        # Number of sources
        source_bonus = min(len(results) * 0.1, 0.3)

        # Source diversity
        unique_sources = len(set(r.get("source", "") for r in results))
        diversity_bonus = min(unique_sources * 0.1, 0.2)

        # Text quality (length and content)
        total_text = sum(len(r.get("snippet", "")) for r in results)
        text_bonus = min(total_text / 1000 * 0.05, 0.2)

        confidence = min(
            base_confidence + source_bonus + diversity_bonus + text_bonus, 0.95
        )

        return round(confidence, 2)

    async def close(self):
        """Clean up resources"""
        await self.search_tool.close()
