"""Evaluation dataset management for RAG systems.

Provides dataset loading, generation, management, and splitting
for RAG evaluation and benchmarking.
"""

import asyncio
import csv
import hashlib
import json
import random
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import structlog

from perfect_rag.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class DatasetFormat(str, Enum):
    """Supported dataset formats."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HUGGINGFACE = "huggingface"


class QuestionType(str, Enum):
    """Types of questions in evaluation datasets."""

    FACTOID = "factoid"  # Simple fact retrieval
    MULTI_HOP = "multi_hop"  # Requires multiple documents
    COMPARISON = "comparison"  # Compare entities
    AGGREGATION = "aggregation"  # Aggregate information
    TEMPORAL = "temporal"  # Time-sensitive questions
    REASONING = "reasoning"  # Requires inference
    UNANSWERABLE = "unanswerable"  # No answer in corpus


@dataclass
class EvaluationSample:
    """A single evaluation sample (Q&A pair)."""

    sample_id: str
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)  # Relevant context passages
    ground_truth_chunks: list[str] = field(default_factory=list)  # Chunk IDs
    ground_truth_docs: list[str] = field(default_factory=list)  # Document IDs
    question_type: QuestionType = QuestionType.FACTOID
    difficulty: str = "medium"  # easy, medium, hard
    domain: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth_chunks": self.ground_truth_chunks,
            "ground_truth_docs": self.ground_truth_docs,
            "question_type": self.question_type.value,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationSample":
        """Create from dictionary."""
        return cls(
            sample_id=data.get("sample_id", secrets.token_hex(8)),
            question=data["question"],
            answer=data.get("answer", ""),
            contexts=data.get("contexts", []),
            ground_truth_chunks=data.get("ground_truth_chunks", []),
            ground_truth_docs=data.get("ground_truth_docs", []),
            question_type=QuestionType(data.get("question_type", "factoid")),
            difficulty=data.get("difficulty", "medium"),
            domain=data.get("domain", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GoldenAnswer:
    """A golden (ground truth) answer with metadata."""

    answer_id: str
    question_id: str
    answer: str
    variants: list[str] = field(default_factory=list)  # Alternative correct answers
    source_chunks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    verified: bool = False
    verified_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer_id": self.answer_id,
            "question_id": self.question_id,
            "answer": self.answer,
            "variants": self.variants,
            "source_chunks": self.source_chunks,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "verified": self.verified,
            "verified_by": self.verified_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenAnswer":
        """Create from dictionary."""
        return cls(
            answer_id=data.get("answer_id", secrets.token_hex(8)),
            question_id=data["question_id"],
            answer=data["answer"],
            variants=data.get("variants", []),
            source_chunks=data.get("source_chunks", []),
            evidence=data.get("evidence", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            created_by=data.get("created_by", ""),
            verified=data.get("verified", False),
            verified_by=data.get("verified_by"),
            metadata=data.get("metadata", {}),
        )


class GoldenAnswerStore:
    """Store and manage ground truth answers."""

    def __init__(
        self,
        surrealdb_client: Any = None,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.surrealdb = surrealdb_client
        self._answers: dict[str, GoldenAnswer] = {}
        self._by_question: dict[str, list[str]] = {}  # question_id -> answer_ids
        self._lock = asyncio.Lock()

    async def add_answer(self, answer: GoldenAnswer) -> None:
        """Add a golden answer."""
        async with self._lock:
            self._answers[answer.answer_id] = answer
            if answer.question_id not in self._by_question:
                self._by_question[answer.question_id] = []
            self._by_question[answer.question_id].append(answer.answer_id)

        if self.surrealdb:
            await self._store_answer(answer)

    async def get_answer(self, answer_id: str) -> GoldenAnswer | None:
        """Get a golden answer by ID."""
        return self._answers.get(answer_id)

    async def get_answers_for_question(
        self,
        question_id: str,
    ) -> list[GoldenAnswer]:
        """Get all golden answers for a question."""
        answer_ids = self._by_question.get(question_id, [])
        return [self._answers[aid] for aid in answer_ids if aid in self._answers]

    async def verify_answer(
        self,
        answer_id: str,
        verified_by: str,
    ) -> GoldenAnswer | None:
        """Mark an answer as verified."""
        answer = self._answers.get(answer_id)
        if answer:
            answer.verified = True
            answer.verified_by = verified_by
            if self.surrealdb:
                await self._store_answer(answer)
        return answer

    async def _store_answer(self, answer: GoldenAnswer) -> None:
        """Store answer in database."""
        if not self.surrealdb:
            return

        try:
            await self.surrealdb.client.query(
                "CREATE golden_answer CONTENT $data",
                {"data": answer.to_dict()},
            )
        except Exception as e:
            logger.error("Failed to store golden answer", error=str(e))

    async def export(self, path: str | Path) -> None:
        """Export all answers to a file."""
        path = Path(path)
        data = [a.to_dict() for a in self._answers.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    async def load(self, path: str | Path) -> int:
        """Load answers from a file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        count = 0
        for item in data:
            answer = GoldenAnswer.from_dict(item)
            await self.add_answer(answer)
            count += 1

        return count


class DatasetLoader:
    """Load evaluation datasets from various formats."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    async def load(
        self,
        source: str | Path,
        format: DatasetFormat | None = None,
        **kwargs,
    ) -> list[EvaluationSample]:
        """Load dataset from source.

        Args:
            source: Path to file or HuggingFace dataset name
            format: Dataset format (auto-detected if None)
            **kwargs: Format-specific options

        Returns:
            List of evaluation samples
        """
        source = str(source)

        # Auto-detect format
        if format is None:
            format = self._detect_format(source)

        logger.info("Loading dataset", source=source, format=format.value)

        if format == DatasetFormat.JSON:
            return await self._load_json(source, **kwargs)
        elif format == DatasetFormat.JSONL:
            return await self._load_jsonl(source, **kwargs)
        elif format == DatasetFormat.CSV:
            return await self._load_csv(source, **kwargs)
        elif format == DatasetFormat.PARQUET:
            return await self._load_parquet(source, **kwargs)
        elif format == DatasetFormat.HUGGINGFACE:
            return await self._load_huggingface(source, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _detect_format(self, source: str) -> DatasetFormat:
        """Detect format from source."""
        source_lower = source.lower()

        if source_lower.endswith(".json"):
            return DatasetFormat.JSON
        elif source_lower.endswith(".jsonl") or source_lower.endswith(".ndjson"):
            return DatasetFormat.JSONL
        elif source_lower.endswith(".csv"):
            return DatasetFormat.CSV
        elif source_lower.endswith(".parquet"):
            return DatasetFormat.PARQUET
        elif "/" in source and not Path(source).exists():
            # Assume HuggingFace dataset
            return DatasetFormat.HUGGINGFACE

        return DatasetFormat.JSON

    async def _load_json(
        self,
        path: str,
        question_field: str = "question",
        answer_field: str = "answer",
        contexts_field: str = "contexts",
        **kwargs,
    ) -> list[EvaluationSample]:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            data = data.get("samples", data.get("data", []))

        samples = []
        for item in data:
            sample = self._parse_sample(
                item,
                question_field,
                answer_field,
                contexts_field,
            )
            samples.append(sample)

        logger.info("Loaded JSON dataset", count=len(samples))
        return samples

    async def _load_jsonl(
        self,
        path: str,
        question_field: str = "question",
        answer_field: str = "answer",
        contexts_field: str = "contexts",
        **kwargs,
    ) -> list[EvaluationSample]:
        """Load from JSON Lines file."""
        samples = []

        with open(path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sample = self._parse_sample(
                        item,
                        question_field,
                        answer_field,
                        contexts_field,
                    )
                    samples.append(sample)

        logger.info("Loaded JSONL dataset", count=len(samples))
        return samples

    async def _load_csv(
        self,
        path: str,
        question_field: str = "question",
        answer_field: str = "answer",
        contexts_field: str = "contexts",
        delimiter: str = ",",
        **kwargs,
    ) -> list[EvaluationSample]:
        """Load from CSV file."""
        samples = []

        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Parse contexts (may be JSON string in CSV)
                contexts = row.get(contexts_field, "")
                if contexts and contexts.startswith("["):
                    try:
                        contexts = json.loads(contexts)
                    except json.JSONDecodeError:
                        contexts = []
                elif contexts:
                    contexts = [contexts]
                else:
                    contexts = []

                sample = EvaluationSample(
                    sample_id=row.get("sample_id", secrets.token_hex(8)),
                    question=row.get(question_field, ""),
                    answer=row.get(answer_field, ""),
                    contexts=contexts,
                    question_type=QuestionType(row.get("question_type", "factoid")),
                    difficulty=row.get("difficulty", "medium"),
                    domain=row.get("domain", ""),
                )
                samples.append(sample)

        logger.info("Loaded CSV dataset", count=len(samples))
        return samples

    async def _load_parquet(
        self,
        path: str,
        question_field: str = "question",
        answer_field: str = "answer",
        **kwargs,
    ) -> list[EvaluationSample]:
        """Load from Parquet file."""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError("pandas and pyarrow required for Parquet support")

        samples = []
        for _, row in df.iterrows():
            sample = EvaluationSample(
                sample_id=str(row.get("sample_id", secrets.token_hex(8))),
                question=str(row.get(question_field, "")),
                answer=str(row.get(answer_field, "")),
                contexts=list(row.get("contexts", [])) if "contexts" in row else [],
            )
            samples.append(sample)

        logger.info("Loaded Parquet dataset", count=len(samples))
        return samples

    async def _load_huggingface(
        self,
        dataset_name: str,
        split: str = "test",
        question_field: str = "question",
        answer_field: str = "answer",
        contexts_field: str = "contexts",
        **kwargs,
    ) -> list[EvaluationSample]:
        """Load from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for HuggingFace support")

        dataset = load_dataset(dataset_name, split=split, **kwargs)

        samples = []
        for item in dataset:
            sample = self._parse_sample(
                dict(item),
                question_field,
                answer_field,
                contexts_field,
            )
            samples.append(sample)

        logger.info(
            "Loaded HuggingFace dataset",
            dataset=dataset_name,
            split=split,
            count=len(samples),
        )
        return samples

    def _parse_sample(
        self,
        item: dict[str, Any],
        question_field: str,
        answer_field: str,
        contexts_field: str,
    ) -> EvaluationSample:
        """Parse a sample from dictionary."""
        contexts = item.get(contexts_field, [])
        if isinstance(contexts, str):
            contexts = [contexts]

        return EvaluationSample(
            sample_id=item.get("sample_id", item.get("id", secrets.token_hex(8))),
            question=item.get(question_field, item.get("query", "")),
            answer=item.get(answer_field, item.get("response", "")),
            contexts=contexts,
            ground_truth_chunks=item.get("ground_truth_chunks", []),
            ground_truth_docs=item.get("ground_truth_docs", []),
            question_type=QuestionType(item.get("question_type", "factoid")),
            difficulty=item.get("difficulty", "medium"),
            domain=item.get("domain", ""),
            metadata=item.get("metadata", {}),
        )


@dataclass
class DatasetSplit:
    """A dataset split with samples and metadata."""

    name: str
    samples: list[EvaluationSample]
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvaluationSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> EvaluationSample:
        return self.samples[idx]


class DatasetSplitter:
    """Split datasets into train/test/validation sets."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int | None = None,
    ):
        if abs(train_ratio + test_ratio + val_ratio - 1.0) > 0.001:
            raise ValueError("Ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def split(
        self,
        samples: list[EvaluationSample],
        stratify_by: str | None = None,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Split samples into train/test/val sets.

        Args:
            samples: List of samples to split
            stratify_by: Field to stratify by (e.g., "question_type", "difficulty")

        Returns:
            Tuple of (train_split, test_split, val_split)
        """
        if self.random_seed is not None:
            random.seed(self.random_seed)

        if stratify_by:
            return self._stratified_split(samples, stratify_by)
        else:
            return self._random_split(samples)

    def _random_split(
        self,
        samples: list[EvaluationSample],
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Random split without stratification."""
        shuffled = samples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        test_end = train_end + int(n * self.test_ratio)

        train_samples = shuffled[:train_end]
        test_samples = shuffled[train_end:test_end]
        val_samples = shuffled[test_end:]

        return (
            DatasetSplit(name="train", samples=train_samples),
            DatasetSplit(name="test", samples=test_samples),
            DatasetSplit(name="validation", samples=val_samples),
        )

    def _stratified_split(
        self,
        samples: list[EvaluationSample],
        stratify_by: str,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Stratified split maintaining class distribution."""
        # Group by stratification key
        groups: dict[str, list[EvaluationSample]] = {}
        for sample in samples:
            key = getattr(sample, stratify_by, "unknown")
            if isinstance(key, Enum):
                key = key.value
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        train_samples = []
        test_samples = []
        val_samples = []

        # Split each group proportionally
        for group_samples in groups.values():
            random.shuffle(group_samples)
            n = len(group_samples)
            train_end = int(n * self.train_ratio)
            test_end = train_end + int(n * self.test_ratio)

            train_samples.extend(group_samples[:train_end])
            test_samples.extend(group_samples[train_end:test_end])
            val_samples.extend(group_samples[test_end:])

        # Shuffle final splits
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        random.shuffle(val_samples)

        return (
            DatasetSplit(name="train", samples=train_samples),
            DatasetSplit(name="test", samples=test_samples),
            DatasetSplit(name="validation", samples=val_samples),
        )


class DatasetGenerator:
    """Generate synthetic evaluation datasets.

    Creates test data for RAG evaluation when real data is unavailable.
    """

    def __init__(
        self,
        llm_gateway: Any = None,
        settings: Settings | None = None,
    ):
        self.llm = llm_gateway
        self.settings = settings or get_settings()

    async def generate_from_documents(
        self,
        documents: list[dict[str, Any]],
        questions_per_doc: int = 3,
        question_types: list[QuestionType] | None = None,
    ) -> list[EvaluationSample]:
        """Generate Q&A pairs from documents.

        Args:
            documents: List of documents with 'content' and 'id' fields
            questions_per_doc: Number of questions per document
            question_types: Types of questions to generate

        Returns:
            List of generated evaluation samples
        """
        if not self.llm:
            raise ValueError("LLM gateway required for generation")

        question_types = question_types or [QuestionType.FACTOID]
        samples = []

        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            doc_id = doc.get("id", doc.get("doc_id", secrets.token_hex(8)))

            for i in range(questions_per_doc):
                q_type = question_types[i % len(question_types)]
                sample = await self._generate_qa_pair(
                    content=content,
                    doc_id=doc_id,
                    question_type=q_type,
                )
                if sample:
                    samples.append(sample)

        logger.info(
            "Generated dataset from documents",
            doc_count=len(documents),
            sample_count=len(samples),
        )

        return samples

    async def _generate_qa_pair(
        self,
        content: str,
        doc_id: str,
        question_type: QuestionType,
    ) -> EvaluationSample | None:
        """Generate a single Q&A pair."""
        type_prompts = {
            QuestionType.FACTOID: "Generate a factual question that can be answered from the text.",
            QuestionType.MULTI_HOP: "Generate a question requiring multiple pieces of information.",
            QuestionType.COMPARISON: "Generate a comparison question if possible.",
            QuestionType.REASONING: "Generate a question requiring inference.",
        }

        prompt = f"""Based on the following text, {type_prompts.get(question_type, type_prompts[QuestionType.FACTOID])}
Then provide the correct answer based only on the text.

Text:
{content[:2000]}

Format your response as:
QUESTION: [your question]
ANSWER: [the answer from the text]"""

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )

            # Parse response
            question = ""
            answer = ""

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("QUESTION:"):
                    question = line[9:].strip()
                elif line.startswith("ANSWER:"):
                    answer = line[7:].strip()

            if question and answer:
                return EvaluationSample(
                    sample_id=secrets.token_hex(8),
                    question=question,
                    answer=answer,
                    contexts=[content[:1000]],
                    ground_truth_docs=[doc_id],
                    question_type=question_type,
                )

        except Exception as e:
            logger.warning("Failed to generate Q&A pair", error=str(e))

        return None

    async def generate_synthetic_dataset(
        self,
        domain: str,
        num_samples: int = 100,
        question_types: list[QuestionType] | None = None,
    ) -> list[EvaluationSample]:
        """Generate fully synthetic evaluation dataset.

        Args:
            domain: Domain for question generation (e.g., "science", "history")
            num_samples: Number of samples to generate
            question_types: Types of questions to include

        Returns:
            List of synthetic evaluation samples
        """
        if not self.llm:
            raise ValueError("LLM gateway required for generation")

        question_types = question_types or [QuestionType.FACTOID]
        samples = []

        for i in range(num_samples):
            q_type = question_types[i % len(question_types)]

            prompt = f"""Generate a {q_type.value} question and answer pair about {domain}.
The question should be answerable and the answer should be factual.

Format:
QUESTION: [question]
ANSWER: [answer]
CONTEXT: [a brief passage that contains the answer]"""

            try:
                response = await self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.8,
                )

                question = ""
                answer = ""
                context = ""

                for line in response.split("\n"):
                    line = line.strip()
                    if line.startswith("QUESTION:"):
                        question = line[9:].strip()
                    elif line.startswith("ANSWER:"):
                        answer = line[7:].strip()
                    elif line.startswith("CONTEXT:"):
                        context = line[8:].strip()

                if question and answer:
                    samples.append(EvaluationSample(
                        sample_id=secrets.token_hex(8),
                        question=question,
                        answer=answer,
                        contexts=[context] if context else [],
                        question_type=q_type,
                        domain=domain,
                        metadata={"synthetic": True},
                    ))

            except Exception as e:
                logger.warning("Failed to generate synthetic sample", error=str(e))

        logger.info(
            "Generated synthetic dataset",
            domain=domain,
            sample_count=len(samples),
        )

        return samples


class EvaluationDataset:
    """Manage evaluation datasets.

    Provides unified interface for loading, storing, and working with
    evaluation datasets.
    """

    def __init__(
        self,
        samples: list[EvaluationSample] | None = None,
        name: str = "evaluation_dataset",
        settings: Settings | None = None,
    ):
        self.samples = samples or []
        self.name = name
        self.settings = settings or get_settings()
        self.loader = DatasetLoader(settings)
        self.splitter = DatasetSplitter()
        self._golden_store = GoldenAnswerStore(settings=settings)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[EvaluationSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> EvaluationSample:
        return self.samples[idx]

    async def load(
        self,
        source: str | Path,
        format: DatasetFormat | None = None,
        **kwargs,
    ) -> int:
        """Load samples from source."""
        new_samples = await self.loader.load(source, format, **kwargs)
        self.samples.extend(new_samples)
        return len(new_samples)

    async def load_golden_answers(self, path: str | Path) -> int:
        """Load golden answers from file."""
        return await self._golden_store.load(path)

    async def save(
        self,
        path: str | Path,
        format: DatasetFormat = DatasetFormat.JSON,
    ) -> None:
        """Save dataset to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [s.to_dict() for s in self.samples]

        if format == DatasetFormat.JSON:
            with open(path, "w") as f:
                json.dump({"name": self.name, "samples": data}, f, indent=2)
        elif format == DatasetFormat.JSONL:
            with open(path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        elif format == DatasetFormat.CSV:
            if data:
                with open(path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        else:
            raise ValueError(f"Unsupported save format: {format}")

        logger.info("Dataset saved", path=str(path), count=len(self.samples))

    def split(
        self,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        stratify_by: str | None = None,
        random_seed: int | None = None,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Split dataset into train/test/validation."""
        self.splitter = DatasetSplitter(
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            random_seed=random_seed,
        )
        return self.splitter.split(self.samples, stratify_by)

    def filter(
        self,
        question_type: QuestionType | None = None,
        difficulty: str | None = None,
        domain: str | None = None,
    ) -> list[EvaluationSample]:
        """Filter samples by criteria."""
        filtered = self.samples

        if question_type:
            filtered = [s for s in filtered if s.question_type == question_type]
        if difficulty:
            filtered = [s for s in filtered if s.difficulty == difficulty]
        if domain:
            filtered = [s for s in filtered if s.domain == domain]

        return filtered

    def sample(self, n: int, random_seed: int | None = None) -> list[EvaluationSample]:
        """Randomly sample n samples."""
        if random_seed:
            random.seed(random_seed)
        return random.sample(self.samples, min(n, len(self.samples)))

    def add_sample(self, sample: EvaluationSample) -> None:
        """Add a sample to the dataset."""
        self.samples.append(sample)

    def remove_sample(self, sample_id: str) -> bool:
        """Remove a sample by ID."""
        for i, sample in enumerate(self.samples):
            if sample.sample_id == sample_id:
                self.samples.pop(i)
                return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.samples),
            "by_question_type": {},
            "by_difficulty": {},
            "by_domain": {},
            "avg_question_length": 0,
            "avg_answer_length": 0,
            "samples_with_contexts": 0,
        }

        if not self.samples:
            return stats

        for sample in self.samples:
            # By question type
            qt = sample.question_type.value
            stats["by_question_type"][qt] = stats["by_question_type"].get(qt, 0) + 1

            # By difficulty
            diff = sample.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # By domain
            if sample.domain:
                stats["by_domain"][sample.domain] = stats["by_domain"].get(sample.domain, 0) + 1

            # Context count
            if sample.contexts:
                stats["samples_with_contexts"] += 1

        # Averages
        stats["avg_question_length"] = sum(len(s.question) for s in self.samples) / len(self.samples)
        stats["avg_answer_length"] = sum(len(s.answer) for s in self.samples) / len(self.samples)

        return stats


# Module-level singleton
_evaluation_dataset: EvaluationDataset | None = None


def get_evaluation_dataset() -> EvaluationDataset:
    """Get or create evaluation dataset."""
    global _evaluation_dataset
    if _evaluation_dataset is None:
        _evaluation_dataset = EvaluationDataset()
    return _evaluation_dataset
