import aiohttp
import asyncio
import re
from typing import Dict, List, Any, Optional
from urllib.parse import quote
from bs4 import BeautifulSoup
from loguru import logger


class GoogleSearchTool:
    """Real Google search implementation with web scraping"""

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
                " Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            self.session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            )

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _scrape_search_results(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Scrape Google search results"""
        await self._ensure_session()

        # Google search URL
        encoded_query = quote(query)
        url = f"https://www.google.com/search?q={encoded_query}&num={num_results}"

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Google search failed with status: {response.status}")
                    return []

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []

                # Find search result containers
                search_divs = soup.find_all("div", class_="g")

                for div in search_divs[:num_results]:
                    try:
                        # Extract title and link
                        title_tag = div.find("h3")
                        if not title_tag:
                            continue

                        title = title_tag.get_text().strip()

                        # Find the link
                        link_tag = div.find("a")
                        if not link_tag or not link_tag.get("href"):
                            continue

                        url = link_tag["href"]

                        # Skip Google internal links
                        if url.startswith("/url?q=") and "google.com" in url:
                            parts = url.split("/url?q=")
                            if len(parts) > 1:
                                actual_url = parts[1].split("&sa=U")[0]
                                url = actual_url

                        # Extract description/snippet
                        snippet_div = div.find("div", class_="VwiC3b")
                        if not snippet_div:
                            # Try alternative snippet selectors
                            snippet_div = div.find("div", class_="s")
                            if snippet_div:
                                # Remove additional elements from snippet
                                for cite in snippet_div.find_all("cite"):
                                    cite.decompose()

                        snippet = snippet_div.get_text().strip() if snippet_div else ""

                        # Clean up snippet
                        snippet = re.sub(r"\s+", " ", snippet)
                        snippet = snippet.replace("...", "")

                        if title and snippet and not url.startswith("#"):
                            results.append(
                                {
                                    "title": title,
                                    "url": url,
                                    "snippet": snippet,
                                    "source": "Google Search",
                                }
                            )

                    except Exception as e:
                        logger.debug(f"Error parsing search result: {e}")
                        continue

                return results

        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []

    async def _scrape_article_content(self, url: str) -> Optional[str]:
        """Scrape the main content from an article URL"""
        if not url.startswith("http"):
            return None

        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Remove script and style elements
                for script in soup(
                    ["script", "style", "nav", "footer", "header", "aside"]
                ):
                    script.decompose()

                # Try to find main content areas
                content_selectors = [
                    "article",
                    "main",
                    '[role="main"]',
                    ".content",
                    ".post-content",
                    ".article-content",
                    ".entry-content",
                    "#content",
                ]

                main_content = None
                for selector in content_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break

                # If no specific content area found, use body
                if not main_content:
                    main_content = soup.find("body")

                if not main_content:
                    return None

                # Get text content
                text = main_content.get_text()

                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                # Limit content length
                if len(text) > 5000:
                    text = text[:5000] + "..."

                return text

        except Exception as e:
            logger.debug(f"Error scraping article content from {url}: {e}")
            return None

    async def search_and_analyze(
        self, query: str, max_results: int = 5
    ) -> Dict[str, Any]:
        """Perform search and analyze results"""
        logger.info(f"Performing Google search for: {query}")

        # Get search results
        search_results = await self._scrape_search_results(query, max_results)

        if not search_results:
            return {
                "success": False,
                "error": "No search results found",
                "query": query,
                "results": [],
            }

        # Enrich results with full content for top 3 results
        enriched_results = []
        for i, result in enumerate(search_results):
            result_data = {
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "source": result["source"],
            }

            # Get full content for top 3 results
            if i < 3:
                try:
                    content = await self._scrape_article_content(result["url"])
                    if content:
                        result_data["full_content"] = content
                except Exception as e:
                    logger.debug(f"Could not get full content for {result['url']}: {e}")

            enriched_results.append(result_data)

        return {
            "success": True,
            "query": query,
            "results": enriched_results,
            "total_results": len(enriched_results),
        }


class ResearchReportGenerator:
    """Generates comprehensive research reports from search results"""

    def __init__(self):
        self.search_tool = GoogleSearchTool()

    async def generate_report(self, topic: str, max_sources: int = 5) -> Dict[str, Any]:
        """Generate a comprehensive research report"""
        logger.info(f"Generating research report for topic: {topic}")

        # Perform initial search
        search_data = await self.search_tool.search_and_analyze(topic, max_sources)

        if not search_data["success"]:
            return {
                "topic": topic,
                "summary": f"Unable to generate research report: {search_data.get('error', 'Unknown error')}",
                "sources": [],
                "key_points": [],
                "confidence": 0.0,
                "error": search_data.get("error"),
            }

        # Extract and synthesize information
        report = self._synthesize_report(topic, search_data["results"])

        # Add additional search if needed for specific aspects
        subqueries = self._generate_subqueries(topic)
        for subquery in subqueries[:2]:  # Limit additional searches
            additional_data = await self.search_tool.search_and_analyze(subquery, 3)
            if additional_data["success"]:
                report["additional_sources"].extend(additional_data["results"])

        # Wait briefly for rate limiting
        await asyncio.sleep(1)

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

        # Extract key information from results
        all_snippets = []
        source_data = []

        for result in results:
            snippet = result.get("snippet", "")
            content = result.get("full_content", snippet)

            all_snippets.append(content)

            source_data.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "snippet": snippet,
                    "type": "web" if "full_content" in result else "snippet",
                }
            )

        # Generate summary (simplified - in production would use LLM for summarization)
        combined_text = " ".join(all_snippets)
        summary = self._extract_summary(combined_text, topic)

        # Extract key points
        key_points = self._extract_key_points(all_snippets)

        # Calculate confidence based on source quality and quantity
        confidence = self._calculate_confidence(results)

        return {
            "topic": topic,
            "summary": summary,
            "sources": source_data,
            "key_points": key_points,
            "confidence": confidence,
            "additional_sources": [],  # For subquery results
        }

    def _generate_subqueries(self, topic: str) -> List[str]:
        """Generate related subqueries for comprehensive research"""
        subqueries = [
            f"{topic} recent developments",
            f"{topic} key findings",
            f"{topic} implications",
        ]
        return subqueries

    def _extract_summary(self, text: str, topic: str) -> str:
        """Extract a summary from the combined text"""
        # Simple extraction - take most relevant sentences
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]

        # Prioritize sentences containing topic-related keywords
        topic_words = topic.lower().split()
        scored_sentences = []

        for sentence in sentences:
            score = 0
            words = sentence.lower().split()

            # Score by topic word matches
            for topic_word in topic_words:
                for word in words:
                    if topic_word in word or word in topic_word:
                        score += 1

            # Prefer sentences with moderate length
            if 50 < len(sentence) < 200:
                score += 1

            scored_sentences.append((sentence, score))

        # Sort by score and take top 3
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]

        return ". ".join(top_sentences) + "."

    def _extract_key_points(self, snippet_list: List[str]) -> List[str]:
        """Extract key points from the snippets"""
        key_points = []

        # Look for sentences with indicators of importance
        importance_indicators = [
            "important",
            "significant",
            "key",
            "crucial",
            "essential",
            "major",
            "primary",
            "main",
            "notable",
            "critical",
        ]

        for text in snippet_list:
            sentences = text.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if 30 < len(sentence) < 150:  # Reasonable length for key point
                    sentence_lower = sentence.lower()
                    for indicator in importance_indicators:
                        if indicator in sentence_lower and sentence not in key_points:
                            key_points.append(sentence)
                            break

        # If no key points found using indicators, extract informative sentences
        if not key_points:
            for text in snippet_list[:5]:  # Limit to first 5 sources
                sentences = [
                    s.strip() for s in text.split(".") if 40 < len(s.strip()) < 200
                ]
                if sentences and len(key_points) < 5:
                    key_points.append(sentences[0])

        return key_points[:5]  # Limit to 5 key points

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on sources"""
        base_confidence = 0.3

        # Add confidence for number of sources
        source_bonus = min(len(results) * 0.1, 0.3)

        # Add confidence for sources with full content
        full_content_sources = sum(1 for r in results if "full_content" in r)
        content_bonus = min(full_content_sources * 0.1, 0.2)

        # Add confidence for source diversity
        unique_domains = len(
            set(r["url"].split("/")[2] for r in results if "://" in r["url"])
        )
        diversity_bonus = min(unique_domains * 0.05, 0.2)

        return min(
            base_confidence + source_bonus + content_bonus + diversity_bonus, 0.95
        )

    async def close(self):
        """Clean up resources"""
        await self.search_tool.close()
