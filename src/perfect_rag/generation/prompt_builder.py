"""Prompt construction for RAG generation."""

from typing import Any

from perfect_rag.config import Settings, get_settings
from perfect_rag.models.query import SourceChunk


class PromptBuilder:
    """Build prompts for RAG generation with proper context formatting."""

    # Default system prompts for different modes
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.
Answer the user's question based on the provided context.
If the context doesn't contain relevant information, acknowledge this and provide your best answer based on general knowledge.
Always cite your sources using [1], [2], etc. when referencing specific information from the context.
Be concise and accurate."""

    STRICT_SYSTEM_PROMPT = """You are a helpful assistant that ONLY uses the provided context to answer questions.
If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question based on the available sources."
Always cite your sources using [1], [2], etc. for every piece of information from the context.
Never make up information not present in the context."""

    CONVERSATIONAL_SYSTEM_PROMPT = """You are a friendly and helpful assistant with access to a knowledge base.
Answer naturally while incorporating relevant information from the provided context.
Use citations [1], [2], etc. when referencing specific facts, but maintain a conversational tone.
If you're unsure about something, say so."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def build_rag_prompt(
        self,
        messages: list[dict[str, str]],
        chunks: list[SourceChunk],
        mode: str = "default",
        custom_system_prompt: str | None = None,
        include_metadata: bool = True,
        task: str | None = None,
    ) -> list[dict[str, str]]:
        """Build prompt with RAG context.

        Args:
            messages: Original conversation messages
            chunks: Retrieved source chunks
            mode: Prompt mode ("default", "strict", "conversational")
            custom_system_prompt: Override system prompt
            include_metadata: Include source metadata in context
            task: Optional task hint (e.g. "mcq" for multiple-choice)

        Returns:
            Messages list with RAG context injected
        """
        # Select system prompt
        if custom_system_prompt:
            base_system = custom_system_prompt
        elif mode == "strict":
            base_system = self.STRICT_SYSTEM_PROMPT
        elif mode == "conversational":
            base_system = self.CONVERSATIONAL_SYSTEM_PROMPT
        else:
            base_system = self.DEFAULT_SYSTEM_PROMPT

        # Build context block
        context_block = self._format_context(chunks, include_metadata, task=task)

        # Combine system prompt with context
        full_system = f"""{base_system}

{context_block}"""

        # Build final messages
        result_messages = []
        has_system = False

        for msg in messages:
            if msg["role"] == "system":
                # Append context to existing system message
                result_messages.append({
                    "role": "system",
                    "content": f"{msg['content']}\n\n{context_block}",
                })
                has_system = True
            else:
                result_messages.append(msg)

        if not has_system:
            # Insert system message at beginning
            result_messages.insert(0, {"role": "system", "content": full_system})

        return result_messages

    def _format_context(
        self,
        chunks: list[SourceChunk],
        include_metadata: bool = True,
        task: str | None = None,
    ) -> str:
        """Format retrieved chunks as context block."""
        if not chunks:
            return "No relevant context found."

        context_parts = ["Retrieved Context:"]
        context_parts.append("=" * 50)

        for i, chunk in enumerate(chunks, 1):
            # Header with source info
            if include_metadata:
                header = f"[{i}] Source: {chunk.doc_title}"
                if chunk.chunk_index > 0:
                    header += f" (section {chunk.chunk_index + 1})"
                context_parts.append(header)
            else:
                context_parts.append(f"[{i}]")

            # Content
            context_parts.append(chunk.content)
            context_parts.append("")  # Empty line between chunks

        context_parts.append("=" * 50)

        if task == "mcq":
            context_parts.append(
                "Instructions: Use the above context only if it is clearly relevant. "
                "Do not include citations or mention sources in the final answer."
            )
        else:
            context_parts.append(
                "Instructions: Use the above context to answer the question. "
                "Cite sources using [1], [2], etc."
            )

        return "\n".join(context_parts)

    def build_multi_step_prompt(
        self,
        query: str,
        chunks: list[SourceChunk],
        step: str = "analyze",
    ) -> list[dict[str, str]]:
        """Build prompts for multi-step reasoning.

        Steps:
        - analyze: Analyze the context for relevant information
        - plan: Plan the answer structure
        - generate: Generate the final answer
        - verify: Verify answer against sources
        """
        context_block = self._format_context(chunks, include_metadata=True)

        if step == "analyze":
            system = """You are an analysis assistant. Your task is to identify the most relevant information from the provided context that helps answer the user's question.

List the key facts and their source numbers [1], [2], etc.
Note any contradictions or gaps in the information."""

        elif step == "plan":
            system = """You are a planning assistant. Based on the analyzed information, create a brief outline of how to answer the question comprehensively.

Include:
- Main points to cover
- Order of presentation
- Sources to cite for each point"""

        elif step == "generate":
            system = """You are a helpful assistant. Generate a comprehensive answer based on the provided context and analysis.

Use citations [1], [2], etc. for all factual claims.
Be accurate and concise."""

        elif step == "verify":
            system = """You are a verification assistant. Check the generated answer against the source context.

Verify:
- All citations are accurate
- No information is fabricated
- The answer fully addresses the question
- Any uncertainties are acknowledged"""

        else:
            system = self.DEFAULT_SYSTEM_PROMPT

        return [
            {"role": "system", "content": f"{system}\n\n{context_block}"},
            {"role": "user", "content": query},
        ]

    def build_hyde_prompt(self, query: str) -> list[dict[str, str]]:
        """Build prompt for HyDE (Hypothetical Document Embeddings)."""
        return [
            {
                "role": "system",
                "content": """Write a short, informative passage (2-3 paragraphs) that would perfectly answer the given question. Write as if you are creating an encyclopedia or textbook entry. Be factual and comprehensive.""",
            },
            {
                "role": "user",
                "content": f"Write a passage answering: {query}",
            },
        ]

    def build_summarization_prompt(
        self,
        chunks: list[SourceChunk],
        focus: str | None = None,
    ) -> list[dict[str, str]]:
        """Build prompt for summarizing retrieved context."""
        context = self._format_context(chunks, include_metadata=True)

        system = """You are a summarization assistant. Create a concise summary of the provided context.

Include:
- Main topics covered
- Key facts and figures
- Source references [1], [2], etc."""

        if focus:
            system += f"\n\nFocus especially on information related to: {focus}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ]

    def format_for_provider(
        self,
        messages: list[dict[str, str]],
        provider: str,
    ) -> list[dict[str, str]]:
        """Format messages for specific LLM provider.

        Handles provider-specific requirements like message ordering,
        role names, and content formatting.
        """
        if provider == "anthropic":
            # Anthropic handles system message separately
            # Already handled in provider implementation
            return messages

        elif provider == "ollama":
            # Ollama uses same format as OpenAI
            return messages

        # Default (OpenAI compatible)
        return messages
