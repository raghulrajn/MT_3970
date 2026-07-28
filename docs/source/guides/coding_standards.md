# Coding Standards

This document defines coding standards for this project. All code must follow these conventions.

1. Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
2. Code should be self-documenting when possible
3. Avoid redundant comments that restate what the code obviously does
4. Use `uv` for dependency management (faster, more reliable than pip)

## Dependency Management

### Use uv for Package Management

This project uses **uv** (<https://github.com/astral-sh/uv>) for dependency management instead of pip.

1. Always use `uv pip` instead of `pip` for installation
2. Keep `pyproject.toml` up to date

### Usage

```bash
# Create an virtual environment With specific python version
uv venv --python 3.11

# Sync dependencies from pyproject.toml
uv sync --all-extras

# Execute code formatting without installing
uvx black your_python_file.py
```

## Comments

It is hard to argue what are "good" or "necessary" comments, therefore guidelines help you to avoid useless comments.

### Google Style Guide Principles

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for comments:

**Key rules:**

1. Comments should explain WHY, not WHAT
2. Avoid redundant comments that restate obvious code
3. Use complete sentences with proper punctuation
4. Keep comments up to date with code changes

### Examples of Good vs Bad Comments

#### Bad: Redundant comments

```python
# Read CSV file
df = pd.read_csv("data.csv")

# Loop through rows
for i in range(len(data)):
    # Add 1 to counter
    counter += 1

# Create empty list
results = []
```

#### Good: Informative comments

```python
# Load cached results to avoid expensive recomputation
df = pd.read_csv("data.csv")

# Process in batches to avoid memory overflow on large datasets
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    results.extend(process_batch(batch))
```

#### Bad: Stating the obvious

```python
class Model:
    def forward(self, x):
        # Apply linear transformation
        x = self.linear(x)
        # Apply ReLU activation
        x = F.relu(x)
        # Return output
        return x
```

#### Good: Explaining non-obvious behavior

```python
class Model:
    def forward(self, x):
        x = self.linear(x)
        # Clamp before ReLU to prevent gradient explosion with large inputs
        x = torch.clamp(x, min=-10, max=10)
        x = F.relu(x)
        return x
```

### When to Comment

#### Do comment

- Complex algorithms or mathematical operations
- Non-obvious design decisions
- Workarounds for bugs or limitations
- Performance optimizations
- References to external documentation or papers

#### Do not comment

- Self-explanatory code
- Standard library operations
- Variable assignments where names are clear
- Simple control flow

## Docstrings

All public functions, classes, and methods must have docstrings.

### Function Docstring Template

```python
def function_name(param1, param2):
    """Brief one-line description (max 80 characters)

    Longer description if needed. Explain what the function does,
    any important behavior, and when to use it.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.
    """
    pass
```

### Class Docstring Template

```python
class ClassName:
    """Brief one-line description.

    Longer description of the class purpose and usage.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.
    """

    def __init__(self, arg1, arg2):
        """Initialize ClassName.

        Args:
            arg1: Description of arg1.
            arg2: Description of arg2.
        """
        pass
```

### Module Docstring Template

```python
"""Brief module description.

Longer description of what this module provides and how to use it.
"""

import ...
```

## Code Formatting

### Automated Formatting

Use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black .
```

Use [isort](https://pycqa.github.io/isort/) for import sorting:

```bash
isort .
```

### Manual Style Guidelines

#### Imports

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import torch
from torch import nn

# Local imports
from src.models.base_model import BaseSurrogateModel
```

#### String quotes

- Use double quotes for strings: `"hello"`
- Use single quotes for dict keys if consistent: `{'key': 'value'}`
- Be consistent within a file

#### Naming conventions

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_leading_underscore`

## Type Hints

Use type hints for function signatures:

```python
def process_data(
    input_path: str,
    batch_size: int = 32,
    shuffle: bool = True
) -> pd.DataFrame:
    """Process input data.

    Args:
        input_path: Path to input file.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data.

    Returns:
        Processed data.
    """
    pass
```

For complex types, use `typing`:

```python
from typing import Dict, List, Optional, Tuple, Union

def get_config() -> Dict[str, Any]:
    pass

def train_model(
    model: nn.Module,
    data: Optional[DataLoader] = None
) -> Tuple[float, Dict[str, float]]:
    pass
```

## Error Handling

### Be specific with exceptions

```python
# Bad
try:
    value = int(user_input)
except:
    pass

# Good
try:
    value = int(user_input)
except ValueError as e:
    logging.error(f"Invalid input: {user_input}. Error: {e}")
    raise
```

### Provide informative error messages

```python
# Bad
if len(data) == 0:
    raise ValueError("Invalid data")

# Good
if len(data) == 0:
    raise ValueError(
        f"Expected non-empty dataset, but received empty array. "
        f"Check that data was loaded correctly from {data_path}."
    )
```

## Testing

### Test Function Naming

```python
def test_<function_name>_<scenario>():
    """Test that <function_name> <expected behavior> when <scenario>."""
    pass
```

Example:

```python
def test_forward_returns_correct_shape():
    """Test that forward() returns correct output shape."""
    model = Model(input_dim=10, output_dim=3)
    x = torch.randn(4, 10)
    y = model(x)
    assert y.shape == (4, 3)
```

### Test Organization

Pytest fixtures let you define reusable test setup. For more information, see the [pytest fixtures documentation](https://docs.pytest.org/en/stable/explanation/fixtures.html).

```python
class TestModelName:
    """Tests for ModelName class."""

    @pytest.fixture
    def model(self):
        """Fixture providing initialized model."""
        return ModelName(config)

    def test_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None

    def test_forward_pass(self, model):
        """Test that forward pass produces expected output."""
        pass
```

## Logging

Use Python's logging module, not print statements:

```python
import logging

log = logging.getLogger(__name__)

# Bad
print(f"Processing {len(data)} samples")
print(f"Error: {error}")

# Good
log.info(f"Processing {len(data)} samples")
log.error(f"Error occurred during processing: {error}", exc_info=True)
```

Log levels:

- `DEBUG`: Detailed information for diagnosing problems
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

## File Organization

### Module Structure

```python
"""Module docstring."""

# Imports
import standard_library
import third_party
from local import modules

# Constants
CONSTANT_VALUE = 42

# Classes
class MyClass:
    pass

# Functions
def my_function():
    pass

# Main execution
if __name__ == "__main__":
    main()
```

### Maximum Line Length

- Code: 100 characters (configured in Black)
- Comments: 80 characters for readability
- Docstrings: 80 characters

Break long lines appropriately (can be automated using black formatter):

```python
# Good
result = some_function_with_long_name(
    first_argument,
    second_argument,
    third_argument,
)

# Good
config = {
    "model_name": "transformer",
    "learning_rate": 0.001,
    "batch_size": 32,
}
```

## Git Commit Messages

### Format

```editor
<type>: <subject>

<body>

<footer>
```

### Types

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Examples

```editor
feat: Add Transformer model implementation

Implement multi-head attention based Transformer model for
surrogate modeling. Includes positional encoding and configurable
number of attention heads.

Closes #123
```

```editor
fix: Correct batch size handling in DataLoader

DataLoader was not respecting configured batch size when dataset
size was not evenly divisible. Now handles remainder batch correctly.
```

## Code Review Checklist

Before pushing code, verify:

- No redundant comments
- Docstrings follow Google style
- Type hints present for public functions
- Code formatted with Black
- Imports sorted with isort
- No print statements (use logging)

**For model evaluation and validation requirements, see [STUDENT_RULES.md](./student_rules.md).**
