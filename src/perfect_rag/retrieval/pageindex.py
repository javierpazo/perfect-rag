"""
PageIndex (VectifyAI) - Reasoning-based retrieval using document tree structure.

PageIndex transforms documents into hierarchical tree structures (like TOC) and uses
LLM reasoning to navigate and find relevant sections - without chunking or vector DB.

This approach achieves 98.7% accuracy on FinanceBench by using tree-based navigation
instead of semantic similarity search.

Reference: https://www.vectify.ai/pageindex
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass
class PageRange:
    """Represents a range of pages in a document."""

    start: int
    end: int
    confidence: float = 1.0
    reasoning: str = ""

    def contains(self, page: int) -> bool:
        """Check if a page number is within this range."""
        return self.start <= page <= self.end

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PageRange":
        """Create from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class TreeNode:
    """Node in the PageIndex tree structure."""

    title: str
    page_start: int
    page_end: int
    level: int = 0
    children: list["TreeNode"] = field(default_factory=list)
    summary: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "level": self.level,
            "children": [c.to_dict() for c in self.children],
            "summary": self.summary,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        """Create from dictionary."""
        return cls(
            title=data["title"],
            page_start=data["page_start"],
            page_end=data["page_end"],
            level=data.get("level", 0),
            children=[cls.from_dict(c) for c in data.get("children", [])],
            summary=data.get("summary", ""),
            keywords=data.get("keywords", []),
        )

    def get_all_page_ranges(self) -> list[tuple[str, int, int]]:
        """Get all page ranges in the tree (title, start, end)."""
        ranges = [(self.title, self.page_start, self.page_end)]
        for child in self.children:
            ranges.extend(child.get_all_page_ranges())
        return ranges

    def find_by_page(self, page: int) -> list["TreeNode"]:
        """Find all nodes that contain a specific page."""
        results = []
        if self.page_start <= page <= self.page_end:
            results.append(self)
        for child in self.children:
            results.extend(child.find_by_page(page))
        return results


class PageIndexRetriever:
    """
    Reasoning-based retrieval using document tree structure.

    Unlike traditional vector search, PageIndex:
    1. Builds a hierarchical TOC from the document
    2. Uses LLM reasoning to navigate the tree
    3. Returns page ranges instead of chunks

    This is more accurate for structured documents (guides, manuals, reports)
    but requires:
    - Document structure (PDFs, books)
    - LLM for reasoning
    - Pre-built tree index
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm_gateway: Any = None,
    ):
        """
        Initialize PageIndex retriever.

        Args:
            settings: Application settings
            llm_gateway: LLM gateway for tree building and search
        """
        self.settings = settings or get_settings()
        self.llm = llm_gateway
        self.tree_cache: dict[str, TreeNode] = {}
        self._tree_path = Path(self.settings.pageindex_tree_path)
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize PageIndex retriever.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        # Ensure tree directory exists
        self._tree_path.mkdir(parents=True, exist_ok=True)

        # Load existing trees into cache
        for tree_file in self._tree_path.glob("*.json"):
            try:
                with open(tree_file, encoding="utf-8") as f:
                    data = json.load(f)
                    doc_id = tree_file.stem
                    self.tree_cache[doc_id] = TreeNode.from_dict(data)
                logger.debug("Loaded PageIndex tree", doc_id=doc_id)
            except Exception as e:
                logger.warning(
                    "Failed to load PageIndex tree",
                    file=str(tree_file),
                    error=str(e),
                )

        self._initialized = True
        logger.info(
            "PageIndex initialized",
            tree_count=len(self.tree_cache),
            tree_path=str(self._tree_path),
        )
        return True

    def should_use_pageindex(self, doc_metadata: dict[str, Any]) -> bool:
        """
        Check if PageIndex should be used for this document.

        Args:
            doc_metadata: Document metadata including page_count, has_structure

        Returns:
            True if PageIndex should be used
        """
        if not self.settings.pageindex_enabled:
            return False

        page_count = doc_metadata.get("page_count", 0)
        if page_count < self.settings.pageindex_min_pages:
            return False

        # Check if document has structure (PDFs, books, reports)
        has_structure = doc_metadata.get("has_structure", True)
        return has_structure

    async def build_tree(
        self,
        pdf_path: str | None = None,
        doc_id: str | None = None,
        pages: list[dict[str, Any]] | None = None,
        toc_data: list[dict[str, Any]] | None = None,
    ) -> TreeNode | None:
        """
        Generate hierarchical TOC tree from document.

        Can build from:
        1. PDF file path (extracts pages and generates TOC via LLM)
        2. Pre-extracted pages list
        3. Existing TOC data

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier
            pages: Pre-extracted pages with text
            toc_data: Existing TOC structure

        Returns:
            Root TreeNode or None if failed
        """
        if not self.llm:
            logger.warning("LLM gateway required for PageIndex tree building")
            return None

        doc_id = doc_id or "unknown"

        # Check cache first
        if doc_id in self.tree_cache:
            logger.debug("Using cached PageIndex tree", doc_id=doc_id)
            return self.tree_cache[doc_id]

        try:
            if toc_data:
                # Build from existing TOC
                tree = self._build_from_toc(toc_data)
            elif pages:
                # Build from extracted pages
                tree = await self._build_from_pages(pages, doc_id)
            elif pdf_path:
                # Build from PDF
                tree = await self._build_from_pdf(pdf_path, doc_id)
            else:
                logger.warning("No input provided for tree building")
                return None

            if tree:
                # Cache and save
                self.tree_cache[doc_id] = tree
                await self._save_tree(tree, doc_id)
                logger.info(
                    "PageIndex tree built",
                    doc_id=doc_id,
                    children=len(tree.children),
                    pages=f"{tree.page_start}-{tree.page_end}",
                )

            return tree

        except Exception as e:
            logger.error("Failed to build PageIndex tree", doc_id=doc_id, error=str(e))
            return None

    def _build_from_toc(self, toc_data: list[dict[str, Any]]) -> TreeNode:
        """Build tree from existing TOC data."""
        # Assume TOC entries have: title, page_start, page_end, level
        root = TreeNode(
            title="Document",
            page_start=1,
            page_end=toc_data[-1].get("page_end", 1) if toc_data else 1,
            level=0,
        )

        stack = [root]
        for entry in toc_data:
            node = TreeNode(
                title=entry.get("title", ""),
                page_start=entry.get("page_start", 1),
                page_end=entry.get("page_end", entry.get("page_start", 1)),
                level=entry.get("level", 1),
            )

            # Find parent based on level
            while len(stack) > 1 and stack[-1].level >= node.level:
                stack.pop()

            stack[-1].children.append(node)
            stack.append(node)

        return root

    async def _build_from_pages(
        self,
        pages: list[dict[str, Any]],
        doc_id: str,
    ) -> TreeNode:
        """Build tree from extracted pages using LLM."""
        total_pages = len(pages)

        # Sample pages for TOC generation (every 5th page + first/last)
        sample_indices = [0] + list(range(4, total_pages, 5)) + [total_pages - 1]
        sample_indices = sorted(set(i for i in sample_indices if i < total_pages))

        sample_pages = [
            f"Page {i + 1}: {pages[i].get('text', '')[:500]}..."
            for i in sample_indices
        ]

        prompt = f"""Analyze this document and create a hierarchical table of contents.

Document has {total_pages} pages. Here are samples:

{chr(10).join(sample_pages)}

Create a JSON structure with sections and subsections. Format:
[
  {{"title": "Section Name", "page_start": 1, "page_end": 10, "level": 1, "summary": "Brief description"}},
  ...
]

Return ONLY valid JSON array, no explanation."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0,
            )

            # Handle response
            response_text = response if isinstance(response, str) else getattr(response, "content", str(response))

            # Parse JSON
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            toc_entries = json.loads(response_text.strip())

            # Build tree
            return self._build_from_toc(toc_entries)

        except Exception as e:
            logger.warning("LLM TOC generation failed, using simple structure", error=str(e))
            # Fallback: simple structure by page ranges
            return TreeNode(
                title="Document",
                page_start=1,
                page_end=total_pages,
                children=[
                    TreeNode(
                        title=f"Section {i + 1}",
                        page_start=i * 10 + 1,
                        page_end=min((i + 1) * 10, total_pages),
                        level=1,
                    )
                    for i in range((total_pages + 9) // 10)
                ],
            )

    async def _build_from_pdf(self, pdf_path: str, doc_id: str) -> TreeNode:
        """Build tree from PDF file."""
        try:
            # Try to use pypdf for extraction
            from pypdf import PdfReader

            reader = PdfReader(pdf_path)
            pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages.append({"page_num": i + 1, "text": text})

            return await self._build_from_pages(pages, doc_id)

        except ImportError:
            logger.warning("pypdf not installed, cannot build tree from PDF")
            return None
        except Exception as e:
            logger.error("Failed to extract PDF", pdf_path=pdf_path, error=str(e))
            return None

    async def _save_tree(self, tree: TreeNode, doc_id: str) -> None:
        """Save tree to disk."""
        tree_file = self._tree_path / f"{doc_id}.json"
        with open(tree_file, "w", encoding="utf-8") as f:
            json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)

    def load_tree(self, doc_id: str) -> TreeNode | None:
        """Load tree from cache or disk."""
        if doc_id in self.tree_cache:
            return self.tree_cache[doc_id]

        tree_file = self._tree_path / f"{doc_id}.json"
        if tree_file.exists():
            try:
                with open(tree_file, encoding="utf-8") as f:
                    data = json.load(f)
                    tree = TreeNode.from_dict(data)
                    self.tree_cache[doc_id] = tree
                    return tree
            except Exception as e:
                logger.warning("Failed to load tree", doc_id=doc_id, error=str(e))

        return None

    async def tree_search(
        self,
        query: str,
        doc_id: str,
        max_depth: int | None = None,
    ) -> list[PageRange]:
        """
        Navigate tree using LLM reasoning to find relevant sections.

        This is the core PageIndex algorithm:
        1. Load document tree
        2. At each level, LLM decides which branches to explore
        3. Return page ranges with reasoning trace

        Args:
            query: User query
            doc_id: Document identifier
            max_depth: Maximum tree depth to explore

        Returns:
            List of PageRange objects with relevant pages
        """
        if not self.llm:
            logger.warning("LLM gateway required for PageIndex tree search")
            return []

        max_depth = max_depth or self.settings.pageindex_max_tree_depth

        # Load tree
        tree = self.load_tree(doc_id)
        if not tree:
            logger.debug("No tree found for document", doc_id=doc_id)
            return []

        try:
            # Navigate tree recursively
            page_ranges = await self._navigate_tree(
                query=query,
                node=tree,
                current_depth=0,
                max_depth=max_depth,
            )

            # Merge overlapping ranges
            merged = self._merge_ranges(page_ranges)

            logger.info(
                "PageIndex tree search complete",
                doc_id=doc_id,
                ranges_found=len(merged),
                query_preview=query[:50],
            )

            return merged

        except Exception as e:
            logger.error("Tree search failed", doc_id=doc_id, error=str(e))
            return []

    async def _navigate_tree(
        self,
        query: str,
        node: TreeNode,
        current_depth: int,
        max_depth: int,
    ) -> list[PageRange]:
        """Recursively navigate tree to find relevant sections."""
        if current_depth >= max_depth or not node.children:
            # Leaf node or max depth - return this range
            return [
                PageRange(
                    start=node.page_start,
                    end=node.page_end,
                    confidence=1.0 - (current_depth * 0.1),  # Penalize deep results
                    reasoning=f"Section: {node.title}",
                )
            ]

        # Ask LLM which children are relevant
        relevant_children = await self._select_relevant_children(query, node)

        if not relevant_children:
            # No relevant children - return current node
            return [
                PageRange(
                    start=node.page_start,
                    end=node.page_end,
                    confidence=0.8,
                    reasoning=f"No specific subsection found for query",
                )
            ]

        # Recursively explore relevant children
        all_ranges = []
        for child_idx in relevant_children:
            child = node.children[child_idx]
            child_ranges = await self._navigate_tree(
                query=query,
                node=child,
                current_depth=current_depth + 1,
                max_depth=max_depth,
            )
            all_ranges.extend(child_ranges)

        return all_ranges

    async def _select_relevant_children(
        self,
        query: str,
        node: TreeNode,
    ) -> list[int]:
        """Use LLM to select which children are relevant to the query."""
        # Format children for LLM
        children_info = [
            f"[{i}] {child.title} (pp. {child.page_start}-{child.page_end})"
            + (f": {child.summary[:100]}..." if child.summary else "")
            for i, child in enumerate(node.children)
        ]

        prompt = f"""Given a query and document sections, select the MOST RELEVANT sections.

Query: {query}

Available sections:
{chr(10).join(children_info)}

Return ONLY the section numbers that are relevant (comma-separated, e.g., "0,2,3").
If none are relevant, return "none".
Maximum 3 sections."""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
            )

            response_text = response if isinstance(response, str) else getattr(response, "content", str(response))
            response_text = response_text.strip().lower()

            if response_text == "none" or not response_text:
                return []

            # Parse indices
            indices = []
            for num in response_text.split(","):
                try:
                    idx = int(num.strip())
                    if 0 <= idx < len(node.children) and idx not in indices:
                        indices.append(idx)
                except ValueError:
                    continue

            return indices[:3]  # Limit to 3

        except Exception as e:
            logger.warning("LLM child selection failed", error=str(e))
            # Fallback: return first child
            return [0] if node.children else []

    def _merge_ranges(self, ranges: list[PageRange]) -> list[PageRange]:
        """Merge overlapping page ranges."""
        if not ranges:
            return []

        # Sort by start page
        sorted_ranges = sorted(ranges, key=lambda r: r.start)
        merged = [sorted_ranges[0]]

        for current in sorted_ranges[1:]:
            previous = merged[-1]

            # Check for overlap or adjacency
            if current.start <= previous.end + 1:
                # Merge
                merged[-1] = PageRange(
                    start=previous.start,
                    end=max(previous.end, current.end),
                    confidence=max(previous.confidence, current.confidence),
                    reasoning=f"{previous.reasoning}; {current.reasoning}",
                )
            else:
                merged.append(current)

        return merged

    def get_metadata_filter(
        self,
        page_ranges: list[PageRange],
    ) -> dict[str, Any]:
        """
        Convert page ranges to a metadata filter for hybrid search.

        Use this to filter Qdrant results to only include chunks
        from the relevant page ranges.

        Args:
            page_ranges: List of PageRange objects

        Returns:
            Metadata filter dict for Qdrant
        """
        if not page_ranges:
            return {}

        # Build filter conditions
        conditions = []
        for pr in page_ranges:
            conditions.append({
                "key": "page_number",
                "range": {
                    "gte": pr.start,
                    "lte": pr.end,
                },
            })

        return {"should": conditions} if len(conditions) > 1 else conditions[0] if conditions else {}


# =============================================================================
# Factory Functions
# =============================================================================

_pageindex_retriever: PageIndexRetriever | None = None


async def get_pageindex_retriever(
    settings: Settings | None = None,
    llm_gateway: Any = None,
) -> PageIndexRetriever:
    """Get or create PageIndex retriever instance."""
    global _pageindex_retriever
    if _pageindex_retriever is None:
        _pageindex_retriever = PageIndexRetriever(
            settings=settings,
            llm_gateway=llm_gateway,
        )
        await _pageindex_retriever.initialize()
    return _pageindex_retriever
