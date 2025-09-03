"""
IPFS integration for document traceability and verification.
"""

import hashlib
import base64
from typing import Optional, Dict, Any
import json
import os


class IPFSManager:
    """Manager for IPFS hash generation and document retrieval."""
    
    def __init__(self, config):
        """Initialize IPFS manager with configuration."""
        self.config = config
        self._document_cache: Dict[str, str] = {}
        self._setup_cache_directory()
    
    def _setup_cache_directory(self):
        """Setup cache directory for IPFS documents."""
        cache_dir = os.path.join(self.config.storage.base_storage_path, "ipfs_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_dir = cache_dir
    
    def generate_ipfs_hash(self, document_content: str) -> str:
        """
        Generate IPFS hash for document content.
        
        This implements a simplified IPFS hash generation that creates
        a deterministic hash based on document content for traceability.
        
        Args:
            document_content: Content of the document
            
        Returns:
            IPFS-style hash string
        """
        if not document_content:
            raise ValueError("Document content cannot be empty")
        
        # Convert content to bytes
        content_bytes = document_content.encode('utf-8')
        
        # Create SHA-256 hash of the content
        sha256_hash = hashlib.sha256(content_bytes).digest()
        
        # Create a simplified IPFS-style hash using base64 encoding
        # Format: Qm + base64 encoded hash (truncated for readability)
        b64_hash = base64.b64encode(sha256_hash).decode('ascii')
        # Remove padding and take first 32 characters for IPFS-like format
        ipfs_hash = 'Qm' + b64_hash.replace('=', '').replace('+', '').replace('/', '')[:32]
        
        # Cache the document content for later retrieval
        self._cache_document(ipfs_hash, document_content)
        
        return ipfs_hash
    
    def retrieve_document(self, ipfs_hash: str) -> str:
        """
        Retrieve full document using IPFS hash.
        
        Args:
            ipfs_hash: IPFS hash of the document
            
        Returns:
            Document content
            
        Raises:
            ValueError: If hash is invalid or document not found
        """
        if not ipfs_hash:
            raise ValueError("IPFS hash cannot be empty")
        
        # Check memory cache first
        if ipfs_hash in self._document_cache:
            return self._document_cache[ipfs_hash]
        
        # Check file cache
        cached_content = self._load_from_cache(ipfs_hash)
        if cached_content is not None:
            self._document_cache[ipfs_hash] = cached_content
            return cached_content
        
        raise ValueError(f"Document with IPFS hash {ipfs_hash} not found")
    
    def validate_hash(self, document_content: str, ipfs_hash: str) -> bool:
        """
        Validate document content against IPFS hash.
        
        Args:
            document_content: Content to validate
            ipfs_hash: Expected IPFS hash
            
        Returns:
            True if content matches hash, False otherwise
        """
        try:
            generated_hash = self.generate_ipfs_hash(document_content)
            return generated_hash == ipfs_hash
        except Exception:
            return False
    
    def _cache_document(self, ipfs_hash: str, content: str) -> None:
        """
        Cache document content both in memory and on disk.
        
        Args:
            ipfs_hash: IPFS hash of the document
            content: Document content to cache
        """
        # Cache in memory
        self._document_cache[ipfs_hash] = content
        
        # Cache on disk
        cache_file = os.path.join(self._cache_dir, f"{ipfs_hash}.json")
        cache_data = {
            'hash': ipfs_hash,
            'content': content,
            'cached_at': str(hash(content))  # Simple integrity check
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception:
            # If caching fails, continue without disk cache
            pass
    
    def _load_from_cache(self, ipfs_hash: str) -> Optional[str]:
        """
        Load document content from disk cache.
        
        Args:
            ipfs_hash: IPFS hash to load
            
        Returns:
            Document content if found, None otherwise
        """
        cache_file = os.path.join(self._cache_dir, f"{ipfs_hash}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Verify integrity
            content = cache_data.get('content', '')
            if cache_data.get('cached_at') == str(hash(content)):
                return content
        except Exception:
            # If loading fails, return None
            pass
        
        return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the IPFS cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        memory_cache_size = len(self._document_cache)
        
        # Count disk cache files
        disk_cache_size = 0
        total_cache_size_bytes = 0
        
        try:
            for filename in os.listdir(self._cache_dir):
                if filename.endswith('.json'):
                    disk_cache_size += 1
                    file_path = os.path.join(self._cache_dir, filename)
                    total_cache_size_bytes += os.path.getsize(file_path)
        except Exception:
            pass
        
        return {
            'memory_cache_entries': memory_cache_size,
            'disk_cache_entries': disk_cache_size,
            'total_cache_size_mb': total_cache_size_bytes / (1024 * 1024),
            'cache_directory': self._cache_dir
        }
    
    def clear_cache(self) -> None:
        """Clear both memory and disk cache."""
        # Clear memory cache
        self._document_cache.clear()
        
        # Clear disk cache
        try:
            for filename in os.listdir(self._cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self._cache_dir, filename)
                    os.remove(file_path)
        except Exception:
            pass