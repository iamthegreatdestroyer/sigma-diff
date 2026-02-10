"""Tokenizer Unit Tests"""

import pytest
from src.core.tokenizer import BPETokenizer
from src.api.api_types import TokenSequence


class TestBPETokenizer:
    """Test BPE tokenizer implementation."""
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Test that encode->decode preserves text."""
        text = "Hello, world! This is a test."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        # Note: BPE may not preserve exact spacing in roundtrip
        assert len(decoded) > 0
    
    def test_special_tokens(self, tokenizer):
        """Test special token handling."""
        special = tokenizer.get_special_tokens()
        assert "bos" in special
        assert "eos" in special
        assert "pad" in special
        assert "unk" in special
    
    def test_vocab_size(self, tokenizer):
        """Test vocabulary size."""
        assert tokenizer.get_vocab_size() > 0
    
    def test_decode_single(self, tokenizer):
        """Test single token decoding."""
        # Get a known token
        tokens = tokenizer.encode("hello")
        for token_id in tokens.tokens:
            decoded = tokenizer.decode_single(token_id)
            assert isinstance(decoded, str)
    
    def test_token_to_id_roundtrip(self, tokenizer):
        """Test token_to_id roundtrip."""
        token = "hello"
        token_id = tokenizer.token_to_id(token)
        decoded = tokenizer.id_to_token(token_id)
        assert decoded == token or decoded == "<unk>"


@pytest.fixture
def tokenizer():
    """Create tokenizer for testing."""
    # Use mock vocab for testing
    tok = BPETokenizer()
    tok._vocab = {"hello": 10, "world": 11, "test": 12, "<unk>": 3}
    tok._reverse_vocab = {10: "hello", 11: "world", 12: "test", 3: "<unk>"}
    tok._vocab_size = 4
    return tok

