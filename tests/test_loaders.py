"""Tests for document loaders."""

import pytest
import tempfile
import json
import csv
from pathlib import Path

from perfect_rag.ingestion.loaders import (
    TextLoader,
    PDFLoader,
    HTMLLoader,
    DocxLoader,
    JSONLoader,
    CSVLoader,
    ExcelLoader,
    load_document,
    LOADERS,
)


class TestTextLoader:
    """Tests for text loader."""

    @pytest.mark.asyncio
    async def test_load_txt_file(self):
        """Test loading .txt file."""
        content = "This is a test document.\nWith multiple lines."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            loader = TextLoader()
            doc = await loader.load(path)

            assert doc.content == content
            assert doc.metadata.mime_type == "text/plain"
            assert doc.id is not None
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_load_md_file(self):
        """Test loading .md file."""
        content = "# Heading\n\nParagraph text."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            path = f.name

        try:
            loader = TextLoader()
            doc = await loader.load(path)

            assert doc.content == content
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_load_with_metadata(self):
        """Test loading with custom metadata."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Content")
            path = f.name

        try:
            loader = TextLoader()
            doc = await loader.load(
                path,
                metadata={
                    "title": "Custom Title",
                    "author": "Test Author",
                    "tags": ["test", "example"],
                },
            )

            assert doc.metadata.title == "Custom Title"
            assert doc.metadata.author == "Test Author"
            assert "test" in doc.metadata.tags
        finally:
            Path(path).unlink()


class TestJSONLoader:
    """Tests for JSON loader."""

    @pytest.mark.asyncio
    async def test_load_json_file(self):
        """Test loading .json file."""
        data = {"key": "value", "number": 42}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            loader = JSONLoader()
            doc = await loader.load(path)

            assert doc.metadata.mime_type == "application/json"
            assert "key" in doc.content
            assert "value" in doc.content
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_load_json_with_content_key(self):
        """Test loading JSON with specific content key."""
        data = {"title": "Doc Title", "text": "The actual content."}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            loader = JSONLoader(content_key="text")
            doc = await loader.load(path)

            assert doc.content == "The actual content."
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_load_jsonl_file(self):
        """Test loading .jsonl file."""
        lines = [
            {"text": "Line 1"},
            {"text": "Line 2"},
            {"text": "Line 3"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")
            path = f.name

        try:
            loader = JSONLoader(content_key="text")
            doc = await loader.load(path)

            assert "Line 1" in doc.content
            assert "Line 2" in doc.content
            assert "Line 3" in doc.content
        finally:
            Path(path).unlink()


class TestCSVLoader:
    """Tests for CSV loader."""

    @pytest.mark.asyncio
    async def test_load_csv_file(self):
        """Test loading .csv file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "city"])
            writer.writerow(["Alice", "30", "NYC"])
            writer.writerow(["Bob", "25", "LA"])
            path = f.name

        try:
            loader = CSVLoader()
            doc = await loader.load(path)

            assert "name" in doc.content
            assert "Alice" in doc.content
            assert "Bob" in doc.content
            assert doc.metadata.custom["row_count"] == 2
            assert "name" in doc.metadata.custom["columns"]
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_load_csv_with_selected_columns(self):
        """Test loading CSV with selected columns."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "age", "city"])
            writer.writerow(["Alice", "30", "NYC"])
            path = f.name

        try:
            loader = CSVLoader(content_columns=["name", "city"])
            doc = await loader.load(path)

            assert "name" in doc.content
            assert "city" in doc.content
            # age should not be prominently featured (only selected cols)
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_load_tsv_file(self):
        """Test loading .tsv file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            f.write("col1\tcol2\n")
            f.write("val1\tval2\n")
            path = f.name

        try:
            loader = CSVLoader()
            doc = await loader.load(path)

            assert "col1" in doc.content
            assert "val1" in doc.content
        finally:
            Path(path).unlink()


class TestHTMLLoader:
    """Tests for HTML loader."""

    @pytest.mark.asyncio
    async def test_load_html_file(self):
        """Test loading .html file."""
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Heading</h1>
            <p>Paragraph content.</p>
            <script>alert('ignored');</script>
        </body>
        </html>
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html)
            path = f.name

        try:
            loader = HTMLLoader()
            doc = await loader.load(path)

            assert "Heading" in doc.content
            assert "Paragraph content" in doc.content
            assert "alert" not in doc.content  # Script removed
            assert doc.metadata.title == "Test Page"
        finally:
            Path(path).unlink()


class TestLoaderRegistry:
    """Tests for loader registry."""

    def test_registry_contains_expected_extensions(self):
        """Test registry has all expected extensions."""
        expected = [".txt", ".md", ".pdf", ".html", ".docx", ".json", ".csv", ".xlsx"]

        for ext in expected:
            assert ext in LOADERS, f"Missing loader for {ext}"

    def test_loader_supports_method(self):
        """Test loader.supports() method."""
        assert TextLoader.supports("document.txt")
        assert TextLoader.supports("readme.md")
        assert not TextLoader.supports("image.png")

        assert PDFLoader.supports("document.pdf")
        assert not PDFLoader.supports("document.txt")


class TestAutoLoader:
    """Tests for automatic loader selection."""

    @pytest.mark.asyncio
    async def test_auto_select_text_loader(self):
        """Test automatic selection of text loader."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Auto test content")
            path = f.name

        try:
            doc = await load_document(path)
            assert doc.content == "Auto test content"
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_auto_select_json_loader(self):
        """Test automatic selection of JSON loader."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test": "data"}, f)
            path = f.name

        try:
            doc = await load_document(path)
            assert doc.metadata.mime_type == "application/json"
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_fallback_to_text_loader(self):
        """Test fallback to text loader for unknown extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Unknown format content")
            path = f.name

        try:
            doc = await load_document(path)
            assert doc.content == "Unknown format content"
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_override_loader_class(self):
        """Test overriding loader class."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            json.dump({"key": "value"}, f)  # Write JSON to .txt file
            path = f.name

        try:
            # Force JSON loader on .txt file
            doc = await load_document(path, loader_class=JSONLoader)
            assert doc.metadata.mime_type == "application/json"
        finally:
            Path(path).unlink()


class TestExcelLoader:
    """Tests for Excel loader."""

    @pytest.mark.asyncio
    async def test_excel_loader_registered(self):
        """Test Excel loader is registered."""
        assert ".xlsx" in LOADERS
        assert ".xls" in LOADERS
        assert LOADERS[".xlsx"] == ExcelLoader

    def test_excel_loader_supports(self):
        """Test Excel loader supports method."""
        assert ExcelLoader.supports("data.xlsx")
        assert ExcelLoader.supports("data.xls")
        assert not ExcelLoader.supports("data.csv")
