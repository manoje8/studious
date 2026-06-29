import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.ingestion.parser.docling_parser import DoclingParser


class Config:
    TABLE_MODE = "accurate"
    DO_TABLES = True
    DO_OCR = False


HTML_FORMATS = {".html", ".htm", ".xhtml"}
OFFICE_FORMATS = {".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}


class TestDoclingParser:
    @pytest.fixture
    def parser(self):
        """Create a DoclingParser instance for testing"""
        return DoclingParser()

    @pytest.fixture
    def mock_pdf_path(self, tmp_path):
        """Create a temporary PDF file path"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()
        return pdf_path

    @pytest.fixture
    def mock_html_path(self, tmp_path):
        """Create a temporary HTML file path"""
        html_path = tmp_path / "test.html"
        html_path.touch()
        return html_path

    @pytest.fixture
    def mock_office_path(self, tmp_path):
        """Create a temporary Office file path"""
        docx_path = tmp_path / "test.docx"
        docx_path.touch()
        return docx_path

    @pytest.fixture
    def mock_converter(self):
        """Create a mock DocumentConverter"""
        with patch("src.ingestion.parser.docling_parser.DocumentConverter") as mock:
            converter_instance = Mock()
            converter_instance.convert.return_value = {"mocked": "result"}
            mock.return_value = converter_instance
            yield mock, converter_instance

    @pytest.fixture
    def mock_docling_result(self):
        """Create a mock Docling result object that returns a valid export_to_dict structure"""
        mock_result = Mock()
        mock_document = Mock()
        doc_dict = {
            "body": {"children": [{"$ref": "#/texts/0"}]},
            "texts": [{"label": "paragraph", "orig": "Hello World"}],
        }
        mock_document.export_to_dict.return_value = doc_dict
        mock_result.document = mock_document
        return mock_result

    # Test Initialization
    def test_initialization(self, parser):
        """Test that DoclingParser initializes correctly"""
        assert hasattr(parser, "_converter_cache")
        assert isinstance(parser._converter_cache, dict)
        assert len(parser._converter_cache) == 0
        assert hasattr(parser, "_converter_cache_lock")
        assert isinstance(parser._converter_cache_lock, type(threading.Lock()))

    # Test check_installation
    def test_check_installation_success(self, parser):
        """Test successful installation check"""
        with patch("docling.document_converter.DocumentConverter", create=True):
            assert parser.check_installation() is True

    def test_check_installation_failure(self, parser):
        """Test failed installation check"""
        # Patch sys.modules to simulate import failure
        with patch.dict("sys.modules", {"docling": None, "docling.document_converter": None}):
            assert parser.check_installation() is False

    # Test _get_converter
    def test_get_converter_creates_new_instance(self, parser):
        """Test that _get_converter creates a new converter instance"""
        with patch("docling.document_converter.DocumentConverter") as mock_converter:
            mock_instance = Mock()
            mock_converter.return_value = mock_instance

            converter = parser._get_converter()

            assert converter == mock_instance
            mock_converter.assert_called_once()

    def test_get_converter_cache_hit(self, parser):
        """Test that _get_converter returns cached converter"""
        with patch("docling.document_converter.DocumentConverter") as mock_converter:
            # First call to create and cache
            first_converter = parser._get_converter()

            # Second call should return cached version
            second_converter = parser._get_converter()

            assert first_converter == second_converter
            # DocumentConverter should only be called once
            assert mock_converter.call_count == 1

    def test_get_converter_different_configs(self, parser):
        """Test that different configurations create different converters"""
        with patch("docling.document_converter.DocumentConverter") as mock_converter:
            # Reset mock before test
            mock_converter.reset_mock()

            # First call with config 1
            with patch("src.ingestion.parser.docling_parser.config") as mock_config:
                mock_config.TABLE_MODE = "accurate"
                mock_config.DO_TABLES = True
                mock_config.DO_OCR = False
                converter1 = parser._get_converter()  # noqa

            # Clear cache to simulate different config
            parser._converter_cache.clear()

            # Second call with config 2
            with patch("src.ingestion.parser.docling_parser.config") as mock_config:
                mock_config.TABLE_MODE = "fast"
                mock_config.DO_TABLES = True
                mock_config.DO_OCR = False
                converter2 = parser._get_converter()  # noqa

            # Should create different converters for different configs
            assert mock_converter.call_count == 2

    def test_get_converter_thread_safety(self, parser):
        """Test that _get_converter is thread-safe"""
        import concurrent.futures

        with patch("docling.document_converter.DocumentConverter") as mock_converter:
            mock_converter.return_value = Mock()

            def get_converter():
                return parser._get_converter()

            # Call get_converter from multiple threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(get_converter) for _ in range(10)]
                results = [f.result() for f in futures]

            # All results should be the same object
            assert all(result == results[0] for result in results)

    # Test parse_doc
    def test_parse_doc_pdf(self, parser, mock_pdf_path):
        """Test parsing a PDF file"""
        with patch.object(parser, "parse_pdf") as mock_parse_pdf:
            mock_parse_pdf.return_value = {"pdf": "result"}

            result = parser.parse_doc(mock_pdf_path)

            assert result == {"pdf": "result"}
            mock_parse_pdf.assert_called_once_with(mock_pdf_path)

    def test_parse_doc_html(self, parser, mock_html_path):
        """Test parsing an HTML file"""
        with patch.object(parser, "parse_html") as mock_parse_html:
            mock_parse_html.return_value = {"html": "result"}

            result = parser.parse_doc(mock_html_path)

            assert result == {"html": "result"}
            mock_parse_html.assert_called_once_with(mock_html_path)

    def test_parse_doc_unsupported_format(self, parser, tmp_path):
        """Test parsing an unsupported file format"""
        txt_path = tmp_path / "test.txt"
        txt_path.touch()

        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse_doc(txt_path)

    def test_parse_doc_file_not_found(self, parser):
        """Test parsing a non-existent file"""
        non_existent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(FileNotFoundError):
            parser.parse_doc(non_existent_path)

    def test_parse_doc_office_format(self, parser, mock_office_path, mock_docling_result):
        """Test parsing an Office format file"""
        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_doc(mock_office_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}
            mock_converter.convert.assert_called_once_with(str(mock_office_path))

    def test_parse_doc_case_insensitive_extension(self, parser, tmp_path):
        """Test that file extension check is case-insensitive"""
        pdf_path = tmp_path / "test.PDF"
        pdf_path.touch()

        with patch.object(parser, "parse_pdf") as mock_parse_pdf:
            mock_parse_pdf.return_value = {"pdf": "result"}

            result = parser.parse_doc(pdf_path)

            assert result == {"pdf": "result"}

    # Test parse_pdf
    def test_parse_pdf_success(self, parser, mock_pdf_path, mock_docling_result):
        """Test successful PDF parsing"""
        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_pdf(mock_pdf_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}
            mock_converter.convert.assert_called_once_with(str(mock_pdf_path))

    def test_parse_pdf_file_not_found(self, parser):
        """Test parsing a non-existent PDF file"""
        non_existent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(FileNotFoundError):
            parser.parse_pdf(non_existent_path)

    def test_parse_pdf_with_path_object(self, parser, tmp_path, mock_docling_result):
        """Test that parse_pdf accepts Path objects"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_pdf(pdf_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}

    def test_parse_pdf_with_string_path(self, parser, tmp_path, mock_docling_result):
        """Test that parse_pdf accepts string paths"""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_pdf(str(pdf_path))

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}

    def test_parse_pdf_converter_error(self, parser, mock_pdf_path):
        """Test error handling in parse_pdf"""
        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.side_effect = Exception("Conversion failed")
            mock_get_converter.return_value = mock_converter

            with pytest.raises(Exception, match="Conversion failed"):
                parser.parse_pdf(mock_pdf_path)

    # Test parse_html
    def test_parse_html_success(self, parser, mock_html_path, mock_docling_result):
        """Test successful HTML parsing"""
        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_html(mock_html_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}
            mock_converter.convert.assert_called_once_with(str(mock_html_path))

    def test_parse_html_file_not_found(self, parser):
        """Test parsing a non-existent HTML file"""
        non_existent_path = Path("/nonexistent/file.html")

        with pytest.raises(FileNotFoundError):
            parser.parse_html(non_existent_path)

    def test_parse_html_unsupported_format(self, parser, tmp_path):
        """Test parsing a non-HTML file with parse_html"""
        txt_path = tmp_path / "test.txt"
        txt_path.touch()

        with pytest.raises(ValueError, match="Unsupported HTML format"):
            parser.parse_html(txt_path)

    def test_parse_html_with_htm_extension(self, parser, tmp_path, mock_docling_result):
        """Test that parse_html accepts .htm files"""
        htm_path = tmp_path / "test.htm"
        htm_path.touch()

        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_html(htm_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}

    def test_parse_html_with_xhtml_extension(self, parser, tmp_path, mock_docling_result):
        """Test that parse_html accepts .xhtml files"""
        xhtml_path = tmp_path / "test.xhtml"
        xhtml_path.touch()

        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_html(xhtml_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}

    # Integration tests (if docling is available)
    @pytest.mark.integration
    def test_parse_pdf_integration(self, parser, tmp_path, mock_docling_result):
        """Integration test for parsing an actual PDF file"""
        # This test requires docling to be installed
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError:
            pytest.skip("Docling not installed")

        # Create a test PDF file path
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        # Mock the converter to avoid actual PDF processing
        with patch.object(parser, "_get_converter") as mock_get_converter:
            mock_converter = Mock()
            mock_converter.convert.return_value = mock_docling_result
            mock_get_converter.return_value = mock_converter

            result = parser.parse_pdf(pdf_path)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == {"type": "text", "text": "Hello World", "page_idx": 0}
            mock_converter.convert.assert_called_once_with(str(pdf_path))

    # Edge cases and additional tests
    def test_parse_doc_with_empty_file_path(self, parser):
        """Test parsing with an empty file path"""
        with pytest.raises(FileNotFoundError):
            parser.parse_doc(Path(""))

    def test_parse_doc_with_directory_path(self, parser, tmp_path):
        """Test parsing with a directory path instead of a file"""
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()

        with pytest.raises(FileNotFoundError):
            parser.parse_doc(dir_path)

    def test_converter_cache_isolation(self):
        """Test that different parser instances have separate caches"""
        parser1 = DoclingParser()
        parser2 = DoclingParser()

        with patch("docling.document_converter.DocumentConverter") as mock_converter:
            # Create unique mock instances for each call
            mock_converter.side_effect = [Mock(), Mock()]

            converter1 = parser1._get_converter()
            converter2 = parser2._get_converter()

            # They should be different instances
            assert converter1 is not converter2

    def test_get_converter_with_do_ocr_options(self, parser):
        """Test converter creation with different OCR options"""
        with patch("src.utils.config") as mock_config:
            mock_config.TABLE_MODE = "fast"
            mock_config.DO_TABLES = False
            mock_config.DO_OCR = True

            with patch("docling.document_converter.DocumentConverter") as mock_converter:
                parser._get_converter()

                # Verify converter was created with correct options
                mock_converter.assert_called_once()
