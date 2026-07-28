# Git Guide

This guide teaches you how to use Git effectively during your thesis project. Even though you're working alone, good Git habits will help you track progress, recover from mistakes, and document your work.

---

## Why Use Git During Your Thesis?

- **Version Control**: Track all changes and experiments
- **Undo Mistakes**: Easily revert bad changes
- **Document Progress**: Your commit history tells the story of your work
- **Backup**: Your work is safe on GitHub/GitLab
- **Portfolio**: Shows supervisors and future employers your development process

---

## Basic Git Workflow

### 1. Check What Changed

Before committing, always check what you modified:

```bash
# See which files changed
git status

# See exact changes in files
git diff

# See changes in a specific file
git diff src/models/my_model.py
```

### 2. Stage Your Changes

Add files you want to commit:

```bash
# Add specific files
git add src/models/my_model.py
git add configs/model/my_model.yaml

# Add all Python files
git add src/**/*.py

# Add everything (use carefully!)
git add .
```

### 3. Commit Your Changes

Create a commit with a meaningful message:

```bash
git commit -m "Add attention mechanism to transformer model"
```

### 4. Push to Remote

Upload your commits to GitHub/GitLab:

```bash
git push
```

---

## How to Write Good Commit Messages

### The Golden Rule

**Answer this question: "What does this commit do?"**

### Good vs Bad Examples

❌ **BAD:**
```bash
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
git commit -m "asdf"
git commit -m "final version"
git commit -m "final version 2"
git commit -m "now it works"
```

✅ **GOOD:**
```bash
git commit -m "Add dropout layers to prevent overfitting"
git commit -m "Fix dimension mismatch in attention layer"
git commit -m "Increase learning rate to 0.001"
git commit -m "Add graph convolution layer to encoder"
git commit -m "Remove unused data preprocessing code"
git commit -m "Update model config with new hyperparameters"
```

### Message Format

Use this template:

```
[Action] [what you changed] [optional: why]

Examples:
Add residual connections to improve gradient flow
Fix batch normalization layer initialization
Remove deprecated data augmentation code
Update README with installation instructions
Refactor model architecture for clarity
```

### Common Action Verbs

- **Add**: New feature, file, function
- **Fix**: Bug fix, correction
- **Update**: Modify existing code
- **Remove**: Delete code, files
- **Refactor**: Restructure without changing behavior
- **Optimize**: Performance improvement
- **Document**: Add comments, docstrings, docs

---

## What to Commit

### ✅ DO Commit

**Source code:**

- Model implementations (`src/models/*.py`)
- Configuration files (`configs/**/*.yaml`)
- Training scripts
- Evaluation scripts
- Tests

**Documentation:**

- README updates
- Code comments
- Experiment notes

**Configuration:**

- `.gitignore`
- `pyproject.toml`

### ❌ DON'T Commit

**Large data files:**

- Datasets (use `DATA_DIR` instead)
- Model checkpoints (`.ckpt` files)
- HDF5 files (`.h5`)

**Generated files:**

- `__pycache__/` directories
- `.pyc` files
- Build artifacts

**Experiment outputs:**

- `experiments/` directory
- `outputs/` directory
- `mlruns/` directory (MLflow tracking data)

**Secrets:**

- API keys
- Passwords
- `.env` files with credentials

**Personal settings:**

- `.vscode/` (except shared settings)
- `.idea/`
- OS-specific files (`.DS_Store`, `Thumbs.db`)

**Temporary files:**

- `*.tmp`
- `*.swp`
- `*~`

Your `.gitignore` should handle most of these automatically.

---

## When to Commit (How Often?)

### The Rule: Commit Logical Units of Work

**Commit when you complete a logical piece of work:**

✅ **Good timing:**

- After implementing a new layer type
- After fixing a specific bug
- After adding a new feature
- After updating documentation
- After successful code refactoring
- **Before** trying something experimental
- At the end of each work session

❌ **Bad timing:**

- After changing 10 different files doing different things
- Only once per week
- Only when everything is "perfect"
- In the middle of implementing a feature

### Practical Example

**Scenario: Adding a new attention mechanism to your model**

```bash
# Step 1: Implement the attention layer
# Edit src/models/my_model.py
git add src/models/my_model.py
git commit -m "Add multi-head attention layer to encoder"

# Step 2: Update config with attention parameters
# Edit configs/model/my_model.yaml
git add configs/model/my_model.yaml
git commit -m "Add attention config parameters (num_heads, dropout)"

# Step 3: Test the changes
# Edit tests/test_my_model.py
git add tests/test_my_model.py
git commit -m "Add unit tests for attention layer"

# Step 4: Update documentation
# Edit README.md
git add README.md
git commit -m "Document attention mechanism in model description"
```

**Notice:** 4 commits, each with a clear purpose. Not one giant "add attention" commit.

---

## Common Git Workflows

### Daily Work Session

```bash
# Start of day: Get latest changes (if working from multiple machines)
git pull

# Work on your code...
# Make changes, test, repeat

# End of session: Save your work
git status                          # Check what changed
git diff                            # Review changes
git add src/models/my_model.py     # Stage changes
git commit -m "Add batch normalization to conv layers"
git push                            # Backup to remote
```

### Trying Experimental Changes

**Before** experimenting, commit your working code:

```bash
# Save current working state
git add .
git commit -m "Working baseline model before trying new optimizer"

# Now experiment freely
# Edit code, try new things...

# If experiment works:
git add .
git commit -m "Switch to AdamW optimizer with improved results"

# If experiment fails:
git reset --hard HEAD  #WARNING: Permanently deletes ALL uncommitted changes!
```

### Fixing Mistakes

**Undo last commit (but keep changes):**

```bash
git reset --soft HEAD~1
```

**Undo last commit and discard changes:**

```bash
git reset --hard HEAD~1
```

**Undo changes in a specific file:**

```bash
git checkout -- src/models/my_model.py
```

**See what you did yesterday:**

```bash
git log --since="yesterday"
```

---

## Pre-commit Hooks: Automatic Code Quality

### What Are Pre-commit Hooks?

Pre-commit hooks are automated checks that run before you create a git commit. They automatically format your code, check for errors, and enforce coding standards. This ensures all code in your repository meets quality standards before it's committed.

### Installation (One-Time Setup)

```bash
# Install pre-commit hooks
uv run pre-commit install

# Test on all files
uv run pre-commit run --all-files
```

### Configured Hooks

This project runs 5 types of checks on every commit:

**1. Black - Code Formatter**
- What: Automatically reformats Python code with consistent spacing and style
- Auto-fixes: Yes
- Example: Changes `def foo(x,y)` to `def foo(x, y)`

**2. isort - Import Sorter**
- What: Organizes import statements alphabetically and by category
- Auto-fixes: Yes
- Example: Groups standard library imports separately from third-party imports

**3. flake8 - Code Linter**
- What: Checks for errors, unused code, style violations, and bugs
- Auto-fixes: No (you must fix manually)
- Common errors: Unused imports (F401), undefined variables (F821), lines too long (E501)

**4. pydocstyle - Docstring Checker**
- What: Verifies docstrings follow Google style guide
- Auto-fixes: No (you must fix manually)
- Checks: Missing docstrings, incorrect format, missing Args/Returns sections

**5. General Checks**
- What: Various file hygiene checks
- Auto-fixes: Yes
- Checks: Trailing whitespace, end-of-file newlines, YAML syntax, merge conflicts

### Workflow with Hooks

**Scenario 1: No issues (hooks pass)**

```bash
git add src/models/my_model.py
git commit -m "Add dropout to prevent overfitting"
# All hooks pass → Commit succeeds
```

**Scenario 2: Auto-fixable issues (Black/isort)**

```bash
git add src/models/my_model.py
git commit -m "Add dropout to prevent overfitting"
# Black: Failed - reformatted file
# isort: Failed - sorted imports
# → Commit blocked, files modified

# Re-stage modified files
git add src/models/my_model.py
git commit -m "Add dropout to prevent overfitting"
# → Commit succeeds
```

**Scenario 3: Manual fixes needed (flake8)**

```bash
git add src/models/my_model.py
git commit -m "Add dropout to prevent overfitting"
# flake8: F401 'torch.nn.functional' imported but unused
# → Commit blocked

# Fix the error (remove unused import)
vim src/models/my_model.py
git add src/models/my_model.py
git commit -m "Add dropout to prevent overfitting"
# → Commit succeeds
```

### Testing Hooks Without Committing

Run hooks manually before committing to catch issues early:

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run hooks on specific files
uv run pre-commit run --files src/models/my_model.py

# Run specific hook only
uv run pre-commit run flake8 --all-files
```

### Common Errors and Solutions

**F401: Imported but unused**
```python
# Problem:
import torch
import numpy as np  # Error: never used

# Solution:
import torch
```

**F841: Variable assigned but never used**
```python
# Problem:
result = expensive_computation()  # Error: never used
return loss

# Solution: Remove it
return loss
```

**E501: Line too long (>100 characters)**
```python
# Problem:
model = MyModel(input_dim=128, hidden_dim=256, output_dim=64, dropout=0.5, activation="relu")

# Solution: Split into multiple lines
model = MyModel(
    input_dim=128,
    hidden_dim=256,
    output_dim=64,
    dropout=0.5,
    activation="relu"
)
```

**D103: Missing docstring**
```python
# Problem:
def calculate_loss(pred, target):
    return torch.mean((pred - target) ** 2)

# Solution: Add Google-style docstring
def calculate_loss(pred, target):
    """Calculate mean squared error loss.

    Args:
        pred: Model predictions tensor.
        target: Ground truth tensor.

    Returns:
        Mean squared error as scalar tensor.
    """
    return torch.mean((pred - target) ** 2)
```

### Error Reference Table

| Error Code | Meaning | Solution |
|------------|---------|----------|
| Black: Failed (files modified) | Code reformatted | Re-stage files and commit again |
| isort: Failed (files modified) | Imports sorted | Re-stage files and commit again |
| F401 | Imported but unused | Remove the unused import |
| F821 | Undefined name | Define variable before using it |
| F841 | Variable assigned but never used | Remove or use the variable |
| E501 | Line too long (>100 chars) | Split into multiple lines |
| D103 | Missing docstring | Add Google-style docstring |

### Why Use Pre-commit Hooks?

- Automatic code formatting saves time
- Catches bugs and errors before they're committed
- Ensures consistent code style across the project
- Helps you learn Python best practices
- Makes code reviews easier for supervisors

**Note:** Hooks are optional but strongly recommended. They prevent common mistakes and improve code quality.

---

## Branching (Optional but Useful)

Branches let you work on experimental features without affecting your main code. If the experiment fails, you can simply delete the branch. If it works, you merge it back.

For an interactive tutorial on git branching, try [Learn Git Branching](https://learngitbranching.js.org/).

For major experiments, use branches:

```bash
# Create branch for experiment
git checkout -b experiment-transformer

# Work on experiment
git add .
git commit -m "Implement transformer architecture"

# If experiment works, merge back:
git checkout main
git merge experiment-transformer

# If experiment fails, just delete branch:
git checkout main
git branch -D experiment-transformer
```

---

## Viewing Your History

```bash
# See all commits
git log

# See commits with file changes
git log --stat

# See commits in one line each
git log --oneline

# See graphical history
git log --graph --oneline --all

# See commits for specific file
git log src/models/my_model.py

# See what changed in a commit
git show <commit-hash>
```

---

## Example: One Week of Good Commits

```bash
Mon 10:00  git commit -m "Initialize project structure"
Mon 14:30  git commit -m "Add base model class with training loop"
Mon 16:00  git commit -m "Create DDACS data loader"

Tue 10:30  git commit -m "Implement MLP baseline model"
Tue 14:00  git commit -m "Add model config for MLP"
Tue 16:30  git commit -m "Fix input dimension mismatch in forward pass"

Wed 11:00  git commit -m "Add convolutional layers to model"
Wed 15:00  git commit -m "Tune learning rate and batch size"
Wed 17:00  git commit -m "Add training logs and visualizations"

Thu 10:00  git commit -m "Implement attention mechanism"
Thu 13:00  git commit -m "Fix attention mask broadcasting issue"
Thu 16:00  git commit -m "Add dropout to attention layers"

Fri 10:30  git commit -m "Run experiments on all datasets"
Fri 14:00  git commit -m "Add evaluation results to README"
Fri 15:30  git commit -m "Update config with best hyperparameters"
```

**Notice:**

- Multiple commits per day
- Each commit has a clear purpose
- Commits are small and focused
- Easy to understand what happened each day

---

## Anti-Patterns to Avoid

### ❌ The "One Giant Commit" Anti-Pattern

```bash
# After 2 weeks of work:
git add .
git commit -m "implement everything"
```

**Problem:** Can't track what changed when, hard to find bugs, impossible to undo specific changes.

### ❌ The "Meaningless Messages" Anti-Pattern

```bash
git commit -m "update"
git commit -m "fix"
git commit -m "changes"
```

**Problem:** Future you won't know what these commits did.

### ❌ The "Commit Everything" Anti-Pattern

```bash
git add .
git commit -m "add stuff"  # Including data files, checkpoints, temp files
```

**Problem:** Repository becomes bloated, slow, hard to work with.

### ❌ The "Never Commit" Anti-Pattern

```bash
# Works for 3 months without committing
# Computer crashes
# All work lost
```

**Problem:** No backup, no history, no way to recover.

---

## Quick Reference

```bash
# Daily commands
git status              # What changed?
git diff                # Show changes
git add <file>          # Stage file
git commit -m "msg"     # Commit with message
git push                # Upload to remote

# Undo commands
git checkout -- <file>  # Discard changes in file
git reset --soft HEAD~1 # Undo last commit, keep changes
git reset --hard HEAD~1 # Undo last commit, discard changes

# View history
git log                 # See commits
git log --oneline       # Brief history
git show <commit>       # See commit details

# Branches
git branch <name>       # Create branch
git checkout <name>     # Switch branch
git checkout -b <name>  # Create and switch
git merge <name>        # Merge branch
```

---

## Summary: The Student Git Manifesto

1. ✅ **Commit often** - Multiple times per day
2. ✅ **Commit logical units** - One feature/fix per commit
3. ✅ **Write clear messages** - Future you will thank you
4. ✅ **Push regularly** - Your work is backed up
5. ✅ **Review before committing** - `git status` and `git diff`
6. ✅ **Don't commit generated files** - Use `.gitignore`
7. ✅ **Use pre-commit hooks** - Automatic quality checks
8. ✅ **Commit before experiments** - Easy to undo if it fails

**Remember:** Git is your safety net and your lab notebook. Use it well!

---

## Getting Help

```bash
# Git documentation
git help <command>

# Examples
git help commit
git help log
git help reset
```

**Need more help?**

- [GitHub Git Guides](https://github.com/git-guides)
- [Git Documentation](https://git-scm.com/doc)
- Ask your supervisor
