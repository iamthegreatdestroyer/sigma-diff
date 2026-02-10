"""
Base Tokenizer Implementation
=============================

Implements TokenizerProtocol for BitNet inference.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from ...api.api_types import TokenSequence
from ...api.interfaces import TokenizerProtocol


class BaseTokenizer(TokenizerProtocol, ABC):
    """
    Abstract base tokenizer implementing TokenizerProtocol.
    """
    
    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self._vocab: Dict[str, int] = {}
        self._reverse_vocab: Dict[int, str] = {}
        self._special_tokens = special_tokens or {
            "bos": 1,
            "eos": 2,
            "pad": 0,
            "unk": 3,
        }
        self._vocab_size = 0
        
        if vocab_path:
            self.load_vocab(vocab_path)
    
    def load_vocab(self, path: Union[str, Path]) -> None:
        """Load vocabulary from file."""
        path = Path(path)
        
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        elif path.suffix in (".txt", ".vocab"):
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    token = line.strip()
                    if token:
                        self._vocab[token] = idx
        else:
            raise ValueError(f"Unsupported vocab format: {path.suffix}")
        
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        self._vocab_size = len(self._vocab)
    
    def save_vocab(self, path: Union[str, Path]) -> None:
        """Save vocabulary to file."""
        path = Path(path)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, indent=2, ensure_ascii=False)
    
    @abstractmethod
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword tokens."""
        ...
    
    @abstractmethod
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert subword tokens back to text."""
        ...
    
    def encode(self, text: str, add_special_tokens: bool = True) -> TokenSequence:
        """
        Encode text to TokenSequence.
        
        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            TokenSequence with token IDs
        """
        # Tokenize to subwords
        subwords = self._tokenize(text)
        
        # Convert to IDs
        token_ids = []
        
        if add_special_tokens and "bos" in self._special_tokens:
            token_ids.append(self._special_tokens["bos"])
        
        for subword in subwords:
            if subword in self._vocab:
                token_ids.append(self._vocab[subword])
            else:
                token_ids.append(self._special_tokens.get("unk", 3))
        
        if add_special_tokens and "eos" in self._special_tokens:
            token_ids.append(self._special_tokens["eos"])
        
        return TokenSequence.from_list(token_ids)
    
    def decode(self, tokens: TokenSequence, skip_special_tokens: bool = True) -> str:
        """
        Decode TokenSequence to text.
        
        Args:
            tokens: TokenSequence to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text string
        """
        subwords = []
        special_ids = set(self._special_tokens.values())
        
        for token_id in tokens.tokens:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            if token_id in self._reverse_vocab:
                subwords.append(self._reverse_vocab[token_id])
            else:
                subwords.append("<unk>")
        
        return self._detokenize(subwords)
    
    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        if token_id in self._reverse_vocab:
            return self._reverse_vocab[token_id]
        return "<unk>"
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mapping."""
        return self._special_tokens.copy()
    
    def add_special_token(self, name: str, token_id: int) -> None:
        """Add a special token."""
        self._special_tokens[name] = token_id
    
    def token_to_id(self, token: str) -> int:
        """Convert token string to ID."""
        return self._vocab.get(token, self._special_tokens.get("unk", 3))
    
    def id_to_token(self, token_id: int) -> str:
        """Convert token ID to string."""
        return self._reverse_vocab.get(token_id, "<unk>")

