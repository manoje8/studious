"""
Unit tests for src/common/utils/hasher.py

Covers:
- sha256_hasher: known input → known output, different inputs → different outputs
- make_yaml_serializable: list, tuple, set, dict, primitive types
- hash_data: default hasher, custom hasher, unhashable-with-fallback
"""

import hashlib

from medici.common.utils.hasher import hash_data, make_yaml_serializable, sha256_hasher

# ---------------------------------------------------------------------------
# sha256_hasher
# ---------------------------------------------------------------------------


class TestSha256Hasher:
    def test_known_input_known_output(self):
        expected = hashlib.sha256(b"hello").hexdigest()
        assert sha256_hasher("hello") == expected

    def test_empty_string(self):
        result = sha256_hasher("")
        assert len(result) == 64

    def test_different_inputs_different_hashes(self):
        assert sha256_hasher("abc") != sha256_hasher("xyz")

    def test_deterministic(self):
        assert sha256_hasher("test") == sha256_hasher("test")

    def test_returns_hex_string(self):
        result = sha256_hasher("sample")
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# make_yaml_serializable
# ---------------------------------------------------------------------------


class TestMakeYamlSerializable:
    def test_list_recursed(self):
        result = make_yaml_serializable([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_tuple_recursed(self):
        result = make_yaml_serializable((1, 2))
        assert result == ["1", "2"]

    def test_set_sorted_and_converted_to_tuple(self):
        result = make_yaml_serializable({3, 1, 2})
        assert isinstance(result, tuple)
        assert result == ("1", "2", "3")

    def test_dict_converted_to_sorted_tuple_of_pairs(self):
        result = make_yaml_serializable({"b": 2, "a": 1})
        assert isinstance(result, tuple)
        # sorted by key: ("a",1), ("b",2) → (("a","1"),("b","2"))
        assert result == (("a", "1"), ("b", "2"))

    def test_primitive_converted_to_string(self):
        assert make_yaml_serializable(42) == "42"
        assert make_yaml_serializable(3.14) == "3.14"
        assert make_yaml_serializable(True) == "True"

    def test_nested_structure(self):
        result = make_yaml_serializable({"key": [1, 2]})
        # dict → tuple of pairs, each pair's value is a list
        assert isinstance(result, tuple)

    def test_none_serializable(self):
        result = make_yaml_serializable(None)
        assert result == "None"


# ---------------------------------------------------------------------------
# hash_data
# ---------------------------------------------------------------------------


class TestHashData:
    def test_hash_data_with_default_hasher(self):
        result = hash_data({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_deterministic_for_same_input(self):
        h1 = hash_data({"key": "value"})
        h2 = hash_data({"key": "value"})
        assert h1 == h2

    def test_hash_data_different_for_different_input(self):
        h1 = hash_data({"key": "a"})
        h2 = hash_data({"key": "b"})
        assert h1 != h2

    def test_hash_data_with_custom_hasher(self):
        custom = lambda s: "CUSTOM_" + s[:5]  # noqa: E731
        result = hash_data("hello", hasher=custom)
        assert result.startswith("CUSTOM_")

    def test_hash_data_with_set_uses_fallback_serializer(self):
        # Sets aren't directly yaml-serializable, should use make_yaml_serializable
        result = hash_data({1, 2, 3})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_data_list(self):
        result = hash_data([1, 2, 3])
        assert isinstance(result, str)

    def test_hash_data_primitive(self):
        result = hash_data(42)
        assert isinstance(result, str)
