# Contributing to Hilbert Quantization

Thank you for your interest in contributing to Hilbert Quantization! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tylerlhess/hilbert-quantization.git
   cd hilbert-quantization
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🧪 Running Tests

### Basic Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=hilbert_quantization --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run specific test file
pytest tests/test_api.py
```

### Test Categories
- **Unit tests**: Fast, isolated component tests
- **Integration tests**: End-to-end workflow tests
- **Benchmark tests**: Performance validation tests (marked as `slow`)

## 📝 Code Style

We use several tools to maintain code quality:

### Formatting
- **Black**: Code formatting
- **isort**: Import sorting

```bash
# Format code
black hilbert_quantization/ tests/
isort hilbert_quantization/ tests/
```

### Linting
- **flake8**: Style guide enforcement
- **mypy**: Type checking

```bash
# Check style
flake8 hilbert_quantization/ tests/
mypy hilbert_quantization/
```

### Pre-commit Hooks
All formatting and linting checks run automatically on commit. To run manually:
```bash
pre-commit run --all-files
```

## 🏗️ Project Structure

```
hilbert_quantization/
├── __init__.py          # Main package exports
├── api.py              # High-level user API
├── config.py           # Configuration management
├── models.py           # Data models and structures
├── exceptions.py       # Custom exceptions
├── core/               # Core algorithms
│   ├── compressor.py   # MPEG-AI compression
│   ├── hilbert_mapper.py # Hilbert curve mapping
│   ├── index_generator.py # Hierarchical indexing
│   ├── pipeline.py     # End-to-end pipeline
│   └── search_engine.py # Similarity search
├── utils/              # Utility functions
└── video_api.py        # Video-enhanced features
```

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:
- Python version and OS
- Hilbert Quantization version
- Minimal code example that reproduces the issue
- Full error traceback
- Expected vs actual behavior

### Feature Requests
For new features, please:
- Describe the use case and motivation
- Provide examples of how the feature would be used
- Consider backward compatibility
- Discuss performance implications

## 💡 Contributing Code

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality:
   - Unit tests for individual components
   - Integration tests for end-to-end workflows
   - Update existing tests if behavior changes

4. **Update documentation**:
   - Add docstrings for new functions/classes
   - Update README if needed
   - Add examples for new features

5. **Run the test suite**:
   ```bash
   pytest
   pre-commit run --all-files
   ```

6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

### Code Guidelines

#### Python Style
- Follow PEP 8 style guide
- Use type hints for all public functions
- Write comprehensive docstrings (Google style)
- Keep functions focused and small
- Use meaningful variable names

#### Testing
- Aim for >90% test coverage
- Write both positive and negative test cases
- Use descriptive test names
- Mock external dependencies
- Test edge cases and error conditions

#### Documentation
- Use Google-style docstrings
- Include examples in docstrings
- Keep README and guides up to date
- Document breaking changes

### Example Contribution

```python
def new_feature(parameters: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Brief description of what this function does.
    
    Args:
        parameters: Description of parameters argument
        threshold: Description of threshold with default value
        
    Returns:
        Dictionary containing results with specific keys
        
    Raises:
        ValueError: When parameters are invalid
        
    Example:
        >>> result = new_feature(np.array([1, 2, 3]))
        >>> print(result['metric'])
        0.75
    """
    if len(parameters) == 0:
        raise ValueError("Parameters cannot be empty")
    
    # Implementation here
    return {"metric": 0.75}
```

## 🔄 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version in `pyproject.toml` and `__init__.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create GitHub release
- [ ] Publish to PyPI

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Acknowledge contributions from others

### Communication
- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for questions and general discussion
- Be patient and helpful in code reviews
- Provide context and examples in discussions

## 📚 Resources

### Documentation
- [API Reference](docs/API_GUIDE.md)
- [Quick Start Guide](docs/QUICK_START_GUIDE.md)
- [Complete Usage Guide](docs/guides/COMPLETE_USAGE_GUIDE.md)

### External Resources
- [NumPy Documentation](https://numpy.org/doc/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)

## ❓ Questions?

If you have questions about contributing:
- Check existing [GitHub Issues](https://github.com/tylerlhess/hilbert-quantization/issues)
- Start a [GitHub Discussion](https://github.com/tylerlhess/hilbert-quantization/discussions)
- Email: tylerlhess@gmail.com

Thank you for contributing to Hilbert Quantization! 🎉
