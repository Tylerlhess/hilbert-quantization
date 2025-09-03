"""
Document type filtering and validation for RAG system.

This module provides functionality to filter documents by type, validate document
integrity, and handle corrupted files during batch processing.
"""

import os
import mimetypes
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Optional dependencies
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    chardet = None
    HAS_CHARDET = False

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    magic = None
    HAS_MAGIC = False

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types for processing."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    RTF = "rtf"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"


@dataclass
class DocumentValidationResult:
    """Result of document validation."""
    is_valid: bool
    document_type: DocumentType
    file_path: str
    file_size: int
    encoding: Optional[str] = None
    mime_type: Optional[str] = None
    error_message: Optional[str] = None
    confidence: float = 1.0  # Confidence in type detection (0-1)
    
    def __post_init__(self):
        """Validate document validation result."""
        if self.file_size < 0:
            raise ValueError("File size must be non-negative")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class DocumentFilterConfig:
    """Configuration for document filtering and validation."""
    allowed_types: Set[DocumentType] = None
    max_file_size_mb: float = 100.0  # 100MB default
    min_file_size_bytes: int = 1  # Minimum 1 byte
    require_text_content: bool = True
    encoding_detection: bool = True
    strict_validation: bool = False  # If True, reject files with low confidence
    confidence_threshold: float = 0.7  # Minimum confidence for type detection
    
    def __post_init__(self):
        """Initialize default allowed types if not specified."""
        if self.allowed_types is None:
            self.allowed_types = {
                DocumentType.TEXT,
                DocumentType.PDF,
                DocumentType.MARKDOWN,
                DocumentType.HTML,
                DocumentType.DOCX,
                DocumentType.JSON
            }
        
        if self.max_file_size_mb <= 0:
            raise ValueError("Max file size must be positive")
        if self.min_file_size_bytes < 0:
            raise ValueError("Min file size must be non-negative")
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")


class DocumentTypeDetector:
    """Detect document types using multiple methods."""
    
    # File extension to document type mapping
    EXTENSION_MAP = {
        '.txt': DocumentType.TEXT,
        '.text': DocumentType.TEXT,
        '.md': DocumentType.MARKDOWN,
        '.markdown': DocumentType.MARKDOWN,
        '.pdf': DocumentType.PDF,
        '.html': DocumentType.HTML,
        '.htm': DocumentType.HTML,
        '.docx': DocumentType.DOCX,
        '.doc': DocumentType.DOCX,
        '.rtf': DocumentType.RTF,
        '.csv': DocumentType.CSV,
        '.json': DocumentType.JSON,
        '.xml': DocumentType.XML,
    }
    
    # MIME type to document type mapping
    MIME_MAP = {
        'text/plain': DocumentType.TEXT,
        'text/markdown': DocumentType.MARKDOWN,
        'application/pdf': DocumentType.PDF,
        'text/html': DocumentType.HTML,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
        'application/msword': DocumentType.DOCX,
        'application/rtf': DocumentType.RTF,
        'text/csv': DocumentType.CSV,
        'application/json': DocumentType.JSON,
        'application/xml': DocumentType.XML,
        'text/xml': DocumentType.XML,
    }
    
    def __init__(self):
        """Initialize document type detector."""
        self.magic_mime = None
        if HAS_MAGIC:
            try:
                # Try to initialize python-magic for better MIME detection
                self.magic_mime = magic.Magic(mime=True)
            except Exception as e:
                logger.warning(f"Could not initialize python-magic: {e}. Using fallback methods.")
        else:
            logger.debug("python-magic not available, using fallback MIME detection")
    
    def detect_type(self, file_path: str) -> Tuple[DocumentType, float]:
        """Detect document type with confidence score.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (DocumentType, confidence_score)
        """
        if not os.path.exists(file_path):
            return DocumentType.UNKNOWN, 0.0
        
        # Method 1: File extension
        ext_type, ext_confidence = self._detect_by_extension(file_path)
        
        # Method 2: MIME type
        mime_type, mime_confidence = self._detect_by_mime_type(file_path)
        
        # Method 3: Content analysis (for text files)
        content_type, content_confidence = self._detect_by_content(file_path)
        
        # Combine results with weighted scoring
        candidates = [
            (ext_type, ext_confidence * 0.4),  # Extension gets 40% weight
            (mime_type, mime_confidence * 0.4),  # MIME type gets 40% weight
            (content_type, content_confidence * 0.2)  # Content gets 20% weight
        ]
        
        # Find the best match
        best_type = DocumentType.UNKNOWN
        best_confidence = 0.0
        
        for doc_type, confidence in candidates:
            if confidence > best_confidence:
                best_type = doc_type
                best_confidence = confidence
        
        return best_type, min(best_confidence, 1.0)
    
    def _detect_by_extension(self, file_path: str) -> Tuple[DocumentType, float]:
        """Detect type by file extension."""
        ext = Path(file_path).suffix.lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext], 0.8
        return DocumentType.UNKNOWN, 0.0
    
    def _detect_by_mime_type(self, file_path: str) -> Tuple[DocumentType, float]:
        """Detect type by MIME type."""
        try:
            # Try python-magic first
            if self.magic_mime:
                mime_type = self.magic_mime.from_file(file_path)
            else:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(file_path)
            
            if mime_type and mime_type in self.MIME_MAP:
                return self.MIME_MAP[mime_type], 0.9
            
            # Check for text/* MIME types
            if mime_type and mime_type.startswith('text/'):
                return DocumentType.TEXT, 0.7
                
        except Exception as e:
            logger.debug(f"MIME type detection failed for {file_path}: {e}")
        
        return DocumentType.UNKNOWN, 0.0
    
    def _detect_by_content(self, file_path: str) -> Tuple[DocumentType, float]:
        """Detect type by analyzing file content."""
        try:
            # Read first few bytes to analyze content
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for PDF signature
            if header.startswith(b'%PDF'):
                return DocumentType.PDF, 0.95
            
            # Check for HTML content
            if b'<html' in header.lower() or b'<!doctype html' in header.lower():
                return DocumentType.HTML, 0.85
            
            # Check for XML content
            if header.startswith(b'<?xml') or b'<xml' in header.lower():
                return DocumentType.XML, 0.85
            
            # Check for JSON content
            if header.strip().startswith(b'{') or header.strip().startswith(b'['):
                try:
                    import json
                    # Try to parse as JSON
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    return DocumentType.JSON, 0.8
                except:
                    pass
            
            # Check for Markdown indicators
            if b'#' in header or b'```' in header or b'[' in header and b'](' in header:
                return DocumentType.MARKDOWN, 0.6
            
            # If it's mostly text, classify as text
            try:
                text_content = header.decode('utf-8', errors='ignore')
                if len(text_content.strip()) > 0:
                    return DocumentType.TEXT, 0.5
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Content analysis failed for {file_path}: {e}")
        
        return DocumentType.UNKNOWN, 0.0


class DocumentValidator:
    """Validate documents for processing in RAG system."""
    
    def __init__(self, config: Optional[DocumentFilterConfig] = None):
        """Initialize document validator.
        
        Args:
            config: Document filter configuration
        """
        self.config = config or DocumentFilterConfig()
        self.type_detector = DocumentTypeDetector()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_document(self, file_path: str) -> DocumentValidationResult:
        """Validate a single document.
        
        Args:
            file_path: Path to the document to validate
            
        Returns:
            DocumentValidationResult: Validation result with details
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return DocumentValidationResult(
                    is_valid=False,
                    document_type=DocumentType.UNKNOWN,
                    file_path=file_path,
                    file_size=0,
                    error_message="File does not exist"
                )
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Check file size limits
            max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return DocumentValidationResult(
                    is_valid=False,
                    document_type=DocumentType.UNKNOWN,
                    file_path=file_path,
                    file_size=file_size,
                    error_message=f"File too large: {file_size} bytes > {max_size_bytes} bytes"
                )
            
            if file_size < self.config.min_file_size_bytes:
                return DocumentValidationResult(
                    is_valid=False,
                    document_type=DocumentType.UNKNOWN,
                    file_path=file_path,
                    file_size=file_size,
                    error_message=f"File too small: {file_size} bytes < {self.config.min_file_size_bytes} bytes"
                )
            
            # Detect document type
            doc_type, confidence = self.type_detector.detect_type(file_path)
            
            # Check if type is allowed
            if doc_type not in self.config.allowed_types:
                return DocumentValidationResult(
                    is_valid=False,
                    document_type=doc_type,
                    file_path=file_path,
                    file_size=file_size,
                    confidence=confidence,
                    error_message=f"Document type {doc_type.value} not allowed"
                )
            
            # Check confidence threshold in strict mode
            if self.config.strict_validation and confidence < self.config.confidence_threshold:
                return DocumentValidationResult(
                    is_valid=False,
                    document_type=doc_type,
                    file_path=file_path,
                    file_size=file_size,
                    confidence=confidence,
                    error_message=f"Low confidence in type detection: {confidence:.2f} < {self.config.confidence_threshold:.2f}"
                )
            
            # Detect encoding for text-based files
            encoding = None
            if self.config.encoding_detection and doc_type in [
                DocumentType.TEXT, DocumentType.MARKDOWN, DocumentType.HTML,
                DocumentType.CSV, DocumentType.JSON, DocumentType.XML
            ]:
                encoding = self._detect_encoding(file_path)
                if encoding is None:
                    return DocumentValidationResult(
                        is_valid=False,
                        document_type=doc_type,
                        file_path=file_path,
                        file_size=file_size,
                        confidence=confidence,
                        error_message="Could not detect text encoding"
                    )
            
            # Validate content readability
            if self.config.require_text_content:
                content_valid, content_error = self._validate_content(file_path, doc_type, encoding)
                if not content_valid:
                    return DocumentValidationResult(
                        is_valid=False,
                        document_type=doc_type,
                        file_path=file_path,
                        file_size=file_size,
                        encoding=encoding,
                        confidence=confidence,
                        error_message=content_error
                    )
            
            # Get MIME type for metadata
            mime_type = None
            try:
                if self.type_detector.magic_mime:
                    mime_type = self.type_detector.magic_mime.from_file(file_path)
                else:
                    mime_type, _ = mimetypes.guess_type(file_path)
            except:
                pass
            
            return DocumentValidationResult(
                is_valid=True,
                document_type=doc_type,
                file_path=file_path,
                file_size=file_size,
                encoding=encoding,
                mime_type=mime_type,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed for {file_path}: {str(e)}")
            return DocumentValidationResult(
                is_valid=False,
                document_type=DocumentType.UNKNOWN,
                file_path=file_path,
                file_size=0,
                error_message=f"Validation error: {str(e)}"
            )
    
    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """Detect text encoding of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected encoding or None if detection failed
        """
        try:
            # Use chardet if available
            if HAS_CHARDET:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(10000)  # Read first 10KB for detection
                
                result = chardet.detect(raw_data)
                if result and result['confidence'] > 0.7:
                    return result['encoding']
            
            # Fallback to common encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        f.read(1000)  # Try to read some content
                    return encoding
                except UnicodeDecodeError:
                    continue
            
        except Exception as e:
            self.logger.debug(f"Encoding detection failed for {file_path}: {e}")
        
        return None
    
    def _validate_content(self, file_path: str, doc_type: DocumentType, encoding: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Validate that document content is readable.
        
        Args:
            file_path: Path to the file
            doc_type: Detected document type
            encoding: Detected encoding
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if doc_type == DocumentType.PDF:
                # For PDF, just check if it's a valid PDF file
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                if not header.startswith(b'%PDF'):
                    return False, "Invalid PDF file format"
                return True, None
            
            elif doc_type in [DocumentType.TEXT, DocumentType.MARKDOWN, DocumentType.HTML, 
                             DocumentType.CSV, DocumentType.XML]:
                # For text-based files, try to read content
                if encoding is None:
                    encoding = 'utf-8'
                
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read(1000)  # Read first 1KB
                
                if len(content.strip()) == 0:
                    return False, "File appears to be empty"
                
                return True, None
            
            elif doc_type == DocumentType.JSON:
                # For JSON, try to parse it
                import json
                with open(file_path, 'r', encoding=encoding or 'utf-8') as f:
                    json.load(f)
                return True, None
            
            else:
                # For other types, assume valid if we got this far
                return True, None
                
        except Exception as e:
            return False, f"Content validation failed: {str(e)}"
    
    def filter_documents(self, file_paths: List[str]) -> Tuple[List[str], List[DocumentValidationResult]]:
        """Filter a list of documents, returning valid ones and validation results.
        
        Args:
            file_paths: List of file paths to filter
            
        Returns:
            Tuple of (valid_file_paths, all_validation_results)
        """
        valid_paths = []
        all_results = []
        
        for file_path in file_paths:
            result = self.validate_document(file_path)
            all_results.append(result)
            
            if result.is_valid:
                valid_paths.append(file_path)
            else:
                self.logger.info(f"Filtered out {file_path}: {result.error_message}")
        
        return valid_paths, all_results
    
    def get_validation_summary(self, results: List[DocumentValidationResult]) -> Dict[str, Any]:
        """Generate a summary of validation results.
        
        Args:
            results: List of validation results
            
        Returns:
            Dictionary with validation summary statistics
        """
        total_files = len(results)
        valid_files = sum(1 for r in results if r.is_valid)
        invalid_files = total_files - valid_files
        
        # Count by document type
        type_counts = {}
        for result in results:
            doc_type = result.document_type.value
            if doc_type not in type_counts:
                type_counts[doc_type] = {'valid': 0, 'invalid': 0}
            
            if result.is_valid:
                type_counts[doc_type]['valid'] += 1
            else:
                type_counts[doc_type]['invalid'] += 1
        
        # Count error types
        error_counts = {}
        for result in results:
            if not result.is_valid and result.error_message:
                error_type = result.error_message.split(':')[0]  # Get error category
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Calculate statistics
        total_size = sum(r.file_size for r in results)
        valid_size = sum(r.file_size for r in results if r.is_valid)
        
        avg_confidence = 0.0
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return {
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'validation_rate': valid_files / max(1, total_files),
            'type_distribution': type_counts,
            'error_distribution': error_counts,
            'total_size_bytes': total_size,
            'valid_size_bytes': valid_size,
            'average_confidence': avg_confidence
        }