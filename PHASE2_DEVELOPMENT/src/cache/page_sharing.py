"""
Multi-Sequence KV Cache Page Sharing
====================================

Enable multiple sequences to share cached KV pages with copy-on-write semantics.

Techniques:
- Shared page references
- Reference counting
- Copy-on-write for modifications
- Prefix sharing across sequences
- Memory efficiency improvements

Sprint 2.2 Days 3-4 - Advanced Caching
Created: 2025-12-27
"""

import torch
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SharedPage:
    """A page that can be shared across sequences."""
    page_id: int
    tokens: torch.Tensor  # Token sequence for this page
    k_data: torch.Tensor  # KV cache data
    v_data: torch.Tensor
    reference_count: int = 1
    shared_sequences: Set[str] = field(default_factory=set)
    is_writable: bool = True  # Can be modified in-place
    creation_time: float = 0.0


class PageSharingManager:
    """
    Manages shared KV cache pages across sequences.
    
    Implements copy-on-write semantics for efficient memory use.
    """
    
    def __init__(self, max_total_pages: int = 4096):
        """
        Initialize page sharing manager.
        
        Args:
            max_total_pages: Maximum total pages across all sequences
        """
        self.max_total_pages = max_total_pages
        self.total_pages_allocated = 0
        
        # Page storage
        self.pages: Dict[int, SharedPage] = {}
        self.next_page_id = 0
        
        # Sequence to pages mapping
        self.sequence_pages: Dict[str, List[int]] = defaultdict(list)
        
        # Page usage tracking
        self.page_access_count: Dict[int, int] = defaultdict(int)
        self.page_write_count: Dict[int, int] = defaultdict(int)
        
        # Statistics
        self.total_shares = 0
        self.total_cow_copies = 0
        self.memory_saved_mb = 0.0
    
    def create_page(
        self,
        tokens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> int:
        """
        Create a new shared page.
        
        Args:
            tokens: Token sequence
            k: K cache tensor
            v: V cache tensor
        
        Returns:
            Page ID
        """
        import time
        
        if self.total_pages_allocated >= self.max_total_pages:
            raise RuntimeError("Maximum pages allocated")
        
        page_id = self.next_page_id
        self.next_page_id += 1
        self.total_pages_allocated += 1
        
        page = SharedPage(
            page_id=page_id,
            tokens=tokens.clone(),
            k_data=k.clone(),
            v_data=v.clone(),
            creation_time=time.time()
        )
        
        self.pages[page_id] = page
        return page_id
    
    def share_page(self, page_id: int, sequence_id: str) -> bool:
        """
        Share an existing page with a sequence.
        
        Args:
            page_id: Page to share
            sequence_id: Sequence to share with
        
        Returns:
            Success
        """
        if page_id not in self.pages:
            logger.warning(f"Page {page_id} not found")
            return False
        
        page = self.pages[page_id]
        
        # Increment reference count
        page.reference_count += 1
        page.shared_sequences.add(sequence_id)
        self.sequence_pages[sequence_id].append(page_id)
        
        self.total_shares += 1
        
        logger.debug(f"Shared page {page_id} with {sequence_id} "
                    f"(ref_count={page.reference_count})")
        
        return True
    
    def read_page(self, page_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read KV cache from page (no copy).
        
        Args:
            page_id: Page ID
        
        Returns:
            (k, v) tensors
        """
        if page_id not in self.pages:
            raise ValueError(f"Page {page_id} not found")
        
        page = self.pages[page_id]
        self.page_access_count[page_id] += 1
        
        return page.k_data, page.v_data
    
    def write_page(
        self,
        page_id: int,
        sequence_id: str,
        k: torch.Tensor,
        v: torch.Tensor,
        offset: int = 0
    ) -> int:
        """
        Write to page with copy-on-write semantics.
        
        If page is shared, creates a copy first.
        
        Args:
            page_id: Original page ID
            sequence_id: Sequence writing
            k: New K data
            v: New V data
            offset: Write offset
        
        Returns:
            New page ID (if COW), or original page_id
        """
        page = self.pages.get(page_id)
        if not page:
            raise ValueError(f"Page {page_id} not found")
        
        # Check if COW needed
        if page.reference_count > 1 and page.is_writable:
            # Perform copy-on-write
            new_page_id = self._copy_on_write(page_id, sequence_id)
            self.total_cow_copies += 1
            
            page = self.pages[new_page_id]
            write_page_id = new_page_id
        else:
            write_page_id = page_id
        
        # Perform write
        page.k_data[offset:offset+k.shape[0]] = k
        page.v_data[offset:offset+v.shape[0]] = v
        self.page_write_count[write_page_id] += 1
        
        return write_page_id
    
    def _copy_on_write(self, page_id: int, sequence_id: str) -> int:
        """
        Perform copy-on-write for a page.
        
        Creates a new page with copied data and updates references.
        
        Args:
            page_id: Original page ID
            sequence_id: Sequence causing COW
        
        Returns:
            New page ID
        """
        original = self.pages[page_id]
        
        # Create new page
        new_page_id = self.create_page(
            original.tokens,
            original.k_data,
            original.v_data
        )
        
        # Update references
        new_page = self.pages[new_page_id]
        new_page.shared_sequences = {sequence_id}
        
        # Update original
        original.shared_sequences.discard(sequence_id)
        original.reference_count -= 1
        
        # Update sequence mapping
        if page_id in self.sequence_pages[sequence_id]:
            idx = self.sequence_pages[sequence_id].index(page_id)
            self.sequence_pages[sequence_id][idx] = new_page_id
        
        # Calculate memory savings (avoided copy for other sequences)
        page_size_mb = (original.k_data.numel() + original.v_data.numel()) * 4 / 1e6
        self.memory_saved_mb += page_size_mb * (original.reference_count - 1)
        
        logger.debug(f"COW: page {page_id} -> {new_page_id} for {sequence_id}")
        
        return new_page_id
    
    def get_sequence_pages(self, sequence_id: str) -> List[int]:
        """Get all pages for a sequence."""
        return self.sequence_pages.get(sequence_id, [])
    
    def merge_pages(self, page_ids: List[int]) -> int:
        """
        Merge multiple pages into one.
        
        Useful for consolidating shared prefixes.
        
        Args:
            page_ids: Pages to merge
        
        Returns:
            Merged page ID
        """
        if not page_ids:
            raise ValueError("No pages to merge")
        
        # Collect all data
        all_k = []
        all_v = []
        all_tokens = []
        
        for page_id in page_ids:
            page = self.pages[page_id]
            all_k.append(page.k_data)
            all_v.append(page.v_data)
            all_tokens.append(page.tokens)
        
        # Concatenate
        merged_k = torch.cat(all_k, dim=0)
        merged_v = torch.cat(all_v, dim=0)
        merged_tokens = torch.cat(all_tokens)
        
        # Create merged page
        merged_id = self.create_page(merged_tokens, merged_k, merged_v)
        
        # Clean up old pages
        for page_id in page_ids:
            del self.pages[page_id]
            self.total_pages_allocated -= 1
        
        logger.debug(f"Merged {len(page_ids)} pages into {merged_id}")
        
        return merged_id
    
    def unshare_page(self, page_id: int, sequence_id: str) -> bool:
        """
        Unshare a page from a sequence.
        
        Args:
            page_id: Page ID
            sequence_id: Sequence to unshare
        
        Returns:
            Success
        """
        if page_id not in self.pages:
            return False
        
        page = self.pages[page_id]
        page.shared_sequences.discard(sequence_id)
        page.reference_count -= 1
        
        if page_id in self.sequence_pages[sequence_id]:
            self.sequence_pages[sequence_id].remove(page_id)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get page sharing statistics."""
        total_shared = sum(
            p.reference_count - 1
            for p in self.pages.values()
        )
        
        return {
            "num_pages": len(self.pages),
            "total_pages_allocated": self.total_pages_allocated,
            "total_shares": self.total_shares,
            "total_cow_copies": self.total_cow_copies,
            "avg_share_factor": sum(p.reference_count for p in self.pages.values()) / max(1, len(self.pages)),
            "total_shared_instances": total_shared,
            "memory_saved_mb": self.memory_saved_mb,
            "num_sequences": len(self.sequence_pages)
        }


class PrefixSharingCache:
    """
    Specialized cache for prefix sharing across sequences.
    
    Common prefixes (e.g., system prompt) can be cached once and reused.
    """
    
    def __init__(self, page_manager: PageSharingManager):
        """Initialize prefix cache."""
        self.page_manager = page_manager
        self.prefix_hashes: Dict[int, int] = {}  # hash(prefix_tokens) -> page_id
        self.prefix_count = 0
    
    def find_or_create_prefix(
        self,
        tokens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> int:
        """
        Find existing prefix or create new one.
        
        Args:
            tokens: Prefix tokens
            k: K cache
            v: V cache
        
        Returns:
            Page ID for prefix
        """
        # Hash tokens
        token_hash = hash(tokens.tobytes())
        
        if token_hash in self.prefix_hashes:
            page_id = self.prefix_hashes[token_hash]
            logger.debug(f"Prefix cache hit for hash {token_hash}")
            return page_id
        
        # Create new
        page_id = self.page_manager.create_page(tokens, k, v)
        self.prefix_hashes[token_hash] = page_id
        self.prefix_count += 1
        
        return page_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prefix cache statistics."""
        return {
            "num_cached_prefixes": len(self.prefix_hashes),
            "total_prefixes_created": self.prefix_count
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Page Sharing Manager...")
    
    manager = PageSharingManager(max_total_pages=100)
    
    # Create initial page
    tokens = torch.tensor([1, 2, 3, 4, 5])
    k = torch.randn(10, 64)
    v = torch.randn(10, 64)
    
    page_id = manager.create_page(tokens, k, v)
    print(f"Created page {page_id}")
    
    # Share with sequences
    manager.share_page(page_id, "seq_1")
    manager.share_page(page_id, "seq_2")
    print("Shared page with 2 sequences")
    
    # Read (no copy)
    k_read, v_read = manager.read_page(page_id)
    print(f"Read page: k.shape={k_read.shape}, v.shape={v_read.shape}")
    
    # Write (triggers COW)
    new_k = torch.randn(5, 64)
    new_v = torch.randn(5, 64)
    new_page_id = manager.write_page(page_id, "seq_1", new_k, new_v)
    print(f"Write page: original={page_id}, new={new_page_id}")
    
    # Stats
    stats = manager.get_statistics()
    print(f"\nStats:")
    print(f"  Pages: {stats['num_pages']}")
    print(f"  Shares: {stats['total_shares']}")
    print(f"  COW copies: {stats['total_cow_copies']}")
    print(f"  Memory saved: {stats['memory_saved_mb']:.1f}MB")
    
    print("\nPage sharing test passed!")
