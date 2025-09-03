"""
Tests for document type filtering and validation functionality.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch

from hilbert_quantization.rag.document_processing.document_validator import (
    DocumentValidator,
    DocumentFilterConfig,
    DocumentType,
    DocumentTypeDetector,
    DocumentValidationResult
)


class TestDocumentType:
    """Test document type enumeration."""
    
    def test_document_types(self):
        """Test document type values."""
        assert DocumentType.TEXT.value == "text"
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.MARKDOWN.value == "markdown"
        assert DocumentType.HTML.value == "html"
        assert DocumentType.DOCX.value == "docx"
        assert DocumentType.RTF.value == "rtf"
        assert DocumentType.CSV.value == "csv"
        assert DocumentType.JSON.value == "json"
        assert DocumentType.XML.value == "xml"
        assert DocumentType.UNKNOWN.value == "unknown"


class TestDocumentValidationResult:
    """Test document validation result model."""
    
    def test_valid_result(self):
        """Test valid validation result."""
        result = DocumentValidationResult(
            is_valid=True,
            document_type=DocumentType.TEXT,
            file_path="test.txt",
            file_size=1024,
            encoding="utf-8",
            confidence=0.9
        )
        
        assert result.is_valid is True
        assert result.document_type == DocumentType.TEXT
        assert result.file_path == "test.txt"
        assert result.file_size == 1024
        assert result.encoding == "utf-8"
        assert result.confidence == 0.9
    
    def test_invalid_file_size(self):
        """Test invalid file size validation."""
        with pytest.raises(ValueError, match="File size must be non-negative"):
            DocumentValidationResult(
                is_valid=True,
                document_type=DocumentType.TEXT,
                file_path="test.txt",
                file_size=-1
            )
    
    def test_invalid_confidence(self):
        """Test invalid confidence validation."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            DocumentValidationResult(
                is_valid=True,
                document_type=DocumentType.TEXT,
                file_path="test.txt",
                file_size=1024,
                confidence=1.5
            )


class TestDocumentFilterConfig:
    """Test document filter configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DocumentFilterConfig()
        
        assert DocumentType.TEXT in config.allowed_types
        assert DocumentType.PDF in config.allowed_types
        assert DocumentType.MARKDOWN in config.allowed_types
        assert config.max_file_size_mb == 100.0
        assert config.min_file_size_bytes == 1
        assert config.require_text_content is True
        assert config.encoding_detection is True
        assert config.strict_validation is False
        assert config.confidence_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom configuration."""
        allowed_types = {DocumentType.TEXT, DocumentType.PDF}
        config = DocumentFilterConfig(
            allowed_types=allowed_types,
            max_file_size_mb=50.0,
            min_file_size_bytes=10,
            strict_validation=True,
            confidence_threshold=0.8
        )
        
        assert config.allowed_types == allowed_types
        assert config.max_file_size_mb == 50.0
        assert config.min_file_size_bytes == 10
        assert config.strict_validation is True
        assert config.confidence_threshold == 0.8
    
    def test_invalid_config(self):
        """Test invalid configuration validation."""
        with pytest.raises(ValueError, match="Max file size must be positive"):
            DocumentFilterConfig(max_file_size_mb=0)
        
        with pytest.raises(ValueError, match="Min file size must be non-negative"):
            DocumentFilterConfig(min_file_size_bytes=-1)
        
        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            DocumentFilterConfig(confidence_threshold=1.5)


class TestDocumentTypeDetector:
    """Test document type detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DocumentTypeDetector()
    
    def create_test_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary test file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    def create_test_binary_file(self, content: bytes, suffix: str = ".bin") -> str:
        """Create a temporary binary test file."""
        temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    def cleanup_file(self, file_path: str):
        """Clean up test file."""
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
    
    def test_detect_by_extension(self):
        """Test type detection by file extension."""
        test_cases = [
            (".txt", DocumentType.TEXT),
            (".md", DocumentType.MARKDOWN),
            (".pdf", DocumentType.PDF),
            (".html", DocumentType.HTML),
            (".json", DocumentType.JSON),
            (".xml", DocumentType.XML),
            (".unknown", DocumentType.UNKNOWN)
        ]
        
        for ext, expected_type in test_cases:
            file_path = self.create_test_file("test content", ext)
            try:
                detected_type, confidence = self.detector._detect_by_extension(file_path)
                if expected_type != DocumentType.UNKNOWN:
                    assert detected_type == expected_type
                    assert confidence > 0
                else:
                    assert confidence == 0
            finally:
                self.cleanup_file(file_path)
    
    def test_detect_by_content_pdf(self):
        """Test PDF detection by content."""
        pdf_content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        file_path = self.create_test_binary_file(pdf_content, ".bin")
        
        try:
            detected_type, confidence = self.detector._detect_by_content(file_path)
            assert detected_type == DocumentType.PDF
            assert confidence > 0.9
        finally:
            self.cleanup_file(file_path)
    
    def test_detect_by_content_html(self):
        """Test HTML detection by content."""
        html_content = "<!DOCTYPE html><html><head><title>Test</title></head></html>"
        file_path = self.create_test_file(html_content, ".bin")
        
        try:
            detected_type, confidence = self.detector._detect_by_content(file_path)
            assert detected_type == DocumentType.HTML
            assert confidence > 0.8
        finally:
            self.cleanup_file(file_path)
    
    def test_detect_by_content_json(self):
        """Test JSON detection by content."""
        json_content = json.dumps({"test": "data", "number": 42})
        file_path = self.create_test_file(json_content, ".bin")
        
        try:
            detected_type, confidence = self.detector._detect_by_content(file_path)
            assert detected_type == DocumentType.JSON
            assert confidence > 0.7
        finally:
            self.cleanup_file(file_path)
    
    def test_detect_by_content_markdown(self):
        """Test Markdown detection by content."""
        markdown_content = "# Test Header\n\nThis is a test with `code` and [link](url)."
        file_path = self.create_test_file(markdown_content, ".bin")
        
        try:
            detected_type, confidence = self.detector._detect_by_content(file_path)
            assert detected_type == DocumentType.MARKDOWN
            assert confidence > 0.5
        finally:
            self.cleanup_file(file_path)
    
    def test_detect_type_integration(self):
        """Test integrated type detection."""
        # Test text file with .txt extension
        text_content = "This is a plain text file."
        file_path = self.create_test_file(text_content, ".txt")
        
        try:
            detected_type, confidence = self.detector.detect_type(file_path)
            assert detected_type == DocumentType.TEXT
            assert confidence > 0.3  # Lower threshold since we don't have all detection libraries
        finally:
            self.cleanup_file(file_path)
    
    def test_detect_nonexistent_file(self):
        """Test detection of non-existent file."""
        detected_type, confidence = self.detector.detect_type("nonexistent.txt")
        assert detected_type == DocumentType.UNKNOWN
        assert confidence == 0.0


class TestDocumentValidator:
    """Test document validator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DocumentFilterConfig(
            allowed_types={DocumentType.TEXT, DocumentType.JSON, DocumentType.MARKDOWN},
            max_file_size_mb=1.0,  # 1MB for testing
            min_file_size_bytes=5,
            strict_validation=False
        )
        self.validator = DocumentValidator(self.config)
    
    def create_test_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary test file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    
    def cleanup_file(self, file_path: str):
        """Clean up test file."""
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
    
    def test_validate_valid_document(self):
        """Test validation of valid document."""
        content = "This is a valid text document with sufficient content."
        file_path = self.create_test_file(content, ".txt")
        
        try:
            result = self.validator.validate_document(file_path)
            
            assert result.is_valid is True
            assert result.document_type == DocumentType.TEXT
            assert result.file_size > 0
            assert result.encoding is not None
            assert result.error_message is None
        finally:
            self.cleanup_file(file_path)
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        result = self.validator.validate_document("nonexistent.txt")
        
        assert result.is_valid is False
        assert result.document_type == DocumentType.UNKNOWN
        assert result.file_size == 0
        assert "does not exist" in result.error_message
    
    def test_validate_file_too_large(self):
        """Test validation of file that's too large."""
        # Create content larger than 1MB limit
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        file_path = self.create_test_file(large_content, ".txt")
        
        try:
            result = self.validator.validate_document(file_path)
            
            assert result.is_valid is False
            assert "too large" in result.error_message
        finally:
            self.cleanup_file(file_path)
    
    def test_validate_file_too_small(self):
        """Test validation of file that's too small."""
        small_content = "x"  # 1 byte, less than 5 byte minimum
        file_path = self.create_test_file(small_content, ".txt")
        
        try:
            result = self.validator.validate_document(file_path)
            
            assert result.is_valid is False
            assert "too small" in result.error_message
        finally:
            self.cleanup_file(file_path)
    
    def test_validate_disallowed_type(self):
        """Test validation of disallowed document type."""
        content = "This is PDF-like content"
        file_path = self.create_test_file(content, ".pdf")
        
        try:
            result = self.validator.validate_document(file_path)
            
            assert result.is_valid is False
            assert "not allowed" in result.error_message
        finally:
            self.cleanup_file(file_path)
    
    def test_validate_empty_file(self):
        """Test validation of empty file."""
        file_path = self.create_test_file("", ".txt")
        
        try:
            result = self.validator.validate_document(file_path)
            
            assert result.is_valid is False
            # Empty file will be caught by size check first
            assert ("empty" in result.error_message or "too small" in result.error_message)
        finally:
            self.cleanup_file(file_path)
    
    def test_validate_json_document(self):
        """Test validation of JSON document."""
        json_content = json.dumps({"test": "data", "valid": True})
        file_path = self.create_test_file(json_content, ".json")
        
        try:
            result = self.validator.validate_document(file_path)
            
            assert result.is_valid is True
            assert result.document_type == DocumentType.JSON
        finally:
            self.cleanup_file(file_path)
    
    def test_filter_documents(self):
        """Test filtering multiple documents."""
        # Create test files
        valid_content = "This is valid content for testing."
        invalid_content = "x"  # Too small
        
        valid_file = self.create_test_file(valid_content, ".txt")
        invalid_file = self.create_test_file(invalid_content, ".txt")
        nonexistent_file = "nonexistent.txt"
        
        try:
            file_paths = [valid_file, invalid_file, nonexistent_file]
            valid_paths, results = self.validator.filter_documents(file_paths)
            
            assert len(valid_paths) == 1
            assert valid_paths[0] == valid_file
            assert len(results) == 3
            
            # Check results
            assert results[0].is_valid is True  # valid_file
            assert results[1].is_valid is False  # invalid_file (too small)
            assert results[2].is_valid is False  # nonexistent_file
            
        finally:
            self.cleanup_file(valid_file)
            self.cleanup_file(invalid_file)
    
    def test_get_validation_summary(self):
        """Test validation summary generation."""
        # Create mock validation results
        results = [
            DocumentValidationResult(
                is_valid=True,
                document_type=DocumentType.TEXT,
                file_path="valid1.txt",
                file_size=1000,
                confidence=0.9
            ),
            DocumentValidationResult(
                is_valid=True,
                document_type=DocumentType.JSON,
                file_path="valid2.json",
                file_size=500,
                confidence=0.8
            ),
            DocumentValidationResult(
                is_valid=False,
                document_type=DocumentType.UNKNOWN,
                file_path="invalid.txt",
                file_size=0,
                error_message="File does not exist",
                confidence=0.0
            )
        ]
        
        summary = self.validator.get_validation_summary(results)
        
        assert summary['total_files'] == 3
        assert summary['valid_files'] == 2
        assert summary['invalid_files'] == 1
        assert summary['validation_rate'] == 2/3
        assert summary['total_size_bytes'] == 1500
        assert summary['valid_size_bytes'] == 1500
        assert summary['average_confidence'] == (0.9 + 0.8 + 0.0) / 3
        
        # Check type distribution
        assert 'text' in summary['type_distribution']
        assert 'json' in summary['type_distribution']
        assert summary['type_distribution']['text']['valid'] == 1
        assert summary['type_distribution']['json']['valid'] == 1
        
        # Check error distribution
        assert 'File does not exist' in summary['error_distribution']
        assert summary['error_distribution']['File does not exist'] == 1
    
    def test_strict_validation_mode(self):
        """Test strict validation mode with confidence threshold."""
        strict_config = DocumentFilterConfig(
            allowed_types={DocumentType.TEXT},
            strict_validation=True,
            confidence_threshold=0.9
        )
        strict_validator = DocumentValidator(strict_config)
        
        # Create a file that might have lower confidence
        content = "Ambiguous content that might not be clearly text."
        file_path = self.create_test_file(content, ".unknown")
        
        try:
            result = strict_validator.validate_document(file_path)
            
            # In strict mode, low confidence should cause rejection
            if result.confidence < 0.9:
                assert result.is_valid is False
                assert "Low confidence" in result.error_message
        finally:
            self.cleanup_file(file_path)


if __name__ == "__main__":
    pytest.main([__file__])