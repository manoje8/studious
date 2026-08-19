"""
Unit tests for src/common/utils/helper.py

Covers:
- separate_content: text-only, mixed, multimodal-only, empty list
- has_internet: successful connection returns True, OSError returns False
- supported_extensions_list: returns a non-empty set including .pdf
- check_env: .env present returns True, absent + non-tty returns True
"""

from unittest.mock import MagicMock, patch

from medici.common.utils.helper import (
    has_internet,
    separate_content,
    supported_extensions_list,
)

# ---------------------------------------------------------------------------
# separate_content
# ---------------------------------------------------------------------------


class TestSeparateContent:
    def test_text_only_returns_joined_text_no_multimodal(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        text, multimodal, _tb = separate_content(content)
        assert "Hello" in text
        assert "World" in text
        assert multimodal == []

    def test_multimodal_only_returns_empty_text(self):
        content = [
            {"type": "image", "data": "base64..."},
            {"type": "table", "rows": []},
        ]
        text, multimodal, _tb = separate_content(content)
        assert text.strip() == ""
        assert len(multimodal) == 2

    def test_mixed_separates_correctly(self):
        content = [
            {"type": "text", "text": "Intro"},
            {"type": "image", "data": "img_data"},
            {"type": "text", "text": "Conclusion"},
        ]
        text, multimodal, _tb = separate_content(content)
        assert "Intro" in text
        assert "Conclusion" in text
        assert len(multimodal) == 1
        assert multimodal[0]["type"] == "image"

    def test_empty_list_returns_empty_results(self):
        text, multimodal, _tb = separate_content([])
        assert text == ""
        assert multimodal == []

    def test_text_items_with_empty_text_excluded(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "   "},
            {"type": "text", "text": "Real content"},
        ]
        text, multimodal, _tb = separate_content(content)
        assert "Real content" in text
        # Empty strings should not appear as double newlines etc
        assert text.strip() == "Real content"

    def test_multimodal_items_have_content_list_index(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "image", "data": "abc"},
        ]
        _, multimodal, _tb = separate_content(content)
        assert "_content_list_index" in multimodal[0]
        assert multimodal[0]["_content_list_index"] == 1  # index in original list

    def test_default_type_is_text(self):
        """Items without 'type' key should be treated as text."""
        content = [{"text": "no type key"}]
        text, multimodal, _tb = separate_content(content)
        assert "no type key" in text
        assert multimodal == []

    def test_text_parts_joined_with_double_newline(self):
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]
        text, _, _tb = separate_content(content)
        assert "\n\n" in text


# ---------------------------------------------------------------------------
# has_internet
# ---------------------------------------------------------------------------


class TestHasInternet:
    def test_successful_connection_returns_true(self):
        with patch("socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock
            mock_sock.connect = MagicMock()
            result = has_internet()
        assert result is True

    def test_os_error_returns_false(self):
        with patch("socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock
            mock_sock.connect.side_effect = OSError("Connection refused")
            result = has_internet()
        assert result is False

    def test_custom_host_and_port(self):
        with patch("socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock
            result = has_internet(host="1.1.1.1", port=80, timeout=1)
        assert result is True  # connect succeeded (mocked)


# ---------------------------------------------------------------------------
# supported_extensions_list
# ---------------------------------------------------------------------------


class TestSupportedExtensionsList:
    def test_returns_set(self):
        result = supported_extensions_list()
        assert isinstance(result, set)

    def test_includes_pdf(self):
        result = supported_extensions_list()
        assert ".pdf" in result

    def test_non_empty(self):
        result = supported_extensions_list()
        assert len(result) > 0

    def test_all_start_with_dot(self):
        result = supported_extensions_list()
        for ext in result:
            assert ext.startswith("."), f"Extension without dot: {ext}"


# ---------------------------------------------------------------------------
# check_env
# ---------------------------------------------------------------------------


class TestCheckEnv:
    def test_env_file_present_returns_true(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value\n")
        with patch("medici.common.utils.helper.os.path.exists", return_value=True):
            from medici.common.utils.helper import check_env

            result = check_env()
        assert result is True

    def test_env_file_absent_non_tty_returns_true(self):
        with (
            patch("medici.common.utils.helper.os.path.exists", return_value=False),
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            from medici.common.utils.helper import check_env

            result = check_env()
        assert result is True
