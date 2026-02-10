"""
BPE Tokenizer Implementation
============================

Byte-Pair Encoding tokenizer for BitNet models.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import json

from .base import BaseTokenizer
from ...api.api_types import TokenSequence


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding tokenizer.
    
    Compatible with GPT-2/LLaMA style BPE vocabularies.
    """
    
    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        merges_path: Optional[Union[str, Path]] = None,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        super().__init__(vocab_path, special_tokens)
        
        self._merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        
        # Pre-tokenization pattern (GPT-2 style)
        # Simple word/number/punctuation splitting
        self._pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^\s\w]+|\s+""",
            re.UNICODE
        )
        
        # Byte encoder for handling all unicode
        self._byte_encoder = self._build_byte_encoder()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        
        if merges_path:
            self.load_merges(merges_path)
    
    def _build_byte_encoder(self) -> Dict[int, str]:
        """Build byte to unicode mapping."""
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}
    
    def load_merges(self, path: Union[str, Path]) -> None:
        """Load BPE merges from file."""
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        
        # Skip header if present
        start_idx = 1 if lines[0].startswith("#") else 0
        
        for idx, line in enumerate(lines[start_idx:]):
            if line.strip():
                parts = line.split()
                if len(parts) == 2:
                    merge = (parts[0], parts[1])
                    self._merges.append(merge)
                    self._merge_ranks[merge] = idx
    
    def _get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Get all adjacent pairs in word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def _bpe(self, token: str) -> List[str]:
        """Apply BPE to a single token."""
        if not token:
            return []
        
        word = list(token)
        
        if len(word) == 1:
            return word
        
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # Find highest priority merge
            best_pair = min(
                pairs,
                key=lambda p: self._merge_ranks.get(p, float("inf"))
            )
            
            if best_pair not in self._merge_ranks:
                break
            
            # Apply merge
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            
            if len(word) == 1:
                break
        
        return word
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE."""
        tokens = []
        
        # Pre-tokenize
        for match in re.finditer(self._pattern, text):
            token = match.group(0)
            
            # Encode to bytes then to unicode
            encoded = "".join(
                self._byte_encoder[b] for b in token.encode("utf-8")
            )
            
            # Apply BPE
            bpe_tokens = self._bpe(encoded)
            tokens.extend(bpe_tokens)
        
        return tokens
    
    def _detokenize(self, tokens: List[str]) -> str:
        """Convert BPE tokens back to text."""
        # Join tokens
        text = "".join(tokens)
        
        # Decode from unicode to bytes
        try:
            decoded_bytes = bytearray(
                self._byte_decoder[c] for c in text
            )
            return decoded_bytes.decode("utf-8", errors="replace")
        except (KeyError, ValueError):
            return text
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        special_tokens: Optional[Dict[str, int]] = None,
    ) -> "BPETokenizer":
        """
        Load tokenizer from pretrained model directory.
        
        Expected files:
        - vocab.json or tokenizer.json
        - merges.txt
        """
        model_path = Path(model_path)
        
        # Find vocab file
        vocab_path = None
        for name in ["vocab.json", "tokenizer.json"]:
            candidate = model_path / name
            if candidate.exists():
                vocab_path = candidate
                break
        
        # Find merges file
        merges_path = model_path / "merges.txt"
        if not merges_path.exists():
            merges_path = None
        
        return cls(
            vocab_path=vocab_path,
            merges_path=merges_path,
            special_tokens=special_tokens,
        )

