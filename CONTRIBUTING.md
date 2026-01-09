# Contributing to Fake Account Detection

First off, thank you for considering contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Environment details** (Python version, OS, package versions)
- **Error messages** and stack traces if applicable

### 💡 Suggesting Features

Feature requests are welcome! Please:

- Use a clear, descriptive title
- Provide a detailed description of the proposed feature
- Explain why this feature would be useful
- Include examples or mockups if possible

### 🔧 Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Fake-account.git
cd Fake-account

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_feature_engineer.py -v
```

### Code Formatting

```bash
# Format code with black
black src/ tests/ scripts/

# Sort imports with isort
isort src/ tests/ scripts/

# Type checking with mypy
mypy src/
```

## Style Guidelines

### Python Style

We follow [PEP 8](https://peps.python.org/pep-0008/) with these additions:

- **Line length**: 88 characters (Black default)
- **Imports**: Use isort for organization
- **Docstrings**: Google style
- **Type hints**: Required for all public functions

### Docstring Example

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """
    Short description of the function.
    
    Longer description if needed, explaining the function's
    behavior in more detail.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is empty.
        
    Example:
        >>> function_name("test", 5)
        True
    """
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**
```
feat: add XGBoost model support
fix: correct language encoding in FeatureEngineer
docs: update API documentation
test: add tests for visualize module
```

## Pull Request Process

1. **Update documentation** if you're changing functionality
2. **Add tests** for new features
3. **Run the test suite** and ensure all tests pass
4. **Update CHANGELOG.md** with your changes
5. **Request review** from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
- [ ] CHANGELOG.md updated

## Project Structure

```
src/
├── __init__.py          # Package exports
├── feature_engineer.py  # Feature extraction
├── train.py             # Model training
└── visualize.py         # Visualization functions

tests/
├── conftest.py          # Pytest fixtures
├── test_feature_engineer.py
└── test_model.py

scripts/
├── compare_models.py    # Model comparison
└── run_inference.py     # Inference script
```

## Questions?

Feel free to open an issue with your question or reach out to the maintainers.

---

Thank you for contributing! 🙏
