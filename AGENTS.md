# AGENTS.md - Agentic Coding Guidelines for nanochat

This file provides guidelines for agentic coding agents working in this repository.

## Project Overview

nanochat is a minimal full-stack LLM training harness. It covers tokenization, pretraining, finetuning, evaluation, inference, and a chat UI. The project is written in Python using PyTorch.

## Build / Test / Development Commands

### Environment Setup
```bash
# Install dependencies with uv (recommended)
uv sync

# Install dev dependencies
uv sync --group dev
```

### Running Tests
```bash
# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_engine.py

# Run a single test function
python -m pytest tests/test_engine.py::test_kv_cache_advance -v

# Run tests excluding slow tests
python -m pytest -m "not slow"

# Run with verbose output
python -m pytest -v
```

### Running the Application
```bash
# Chat Web UI
python -m scripts.chat_web

# Chat CLI
python -m scripts.chat_cli

# Base model training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12

# SFT training
python -m scripts.chat_sft
```

### Code Quality
```bash
# Check Python syntax
python -m py_compile <file.py>

# Type checking (if mypy installed)
python -m mypy nanochat/
```

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports (torch, fastapi, etc.)
- Local imports (nanochat.* modules)
- Use explicit imports, avoid `import *`

Example:
```python
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `GPTConfig`, `CausalSelfAttention`)
- **Functions/variables**: snake_case (e.g., `get_dist_info`, `kv_cache`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_SEQ_LEN`)
- **Private methods/variables**: prefix with underscore (e.g., `_init_weights`)

### Type Annotations
- Use type hints for function arguments and return values
- Prefer `typing` module (List, Dict, Optional, etc.) for compatibility
- Use dataclasses for configuration objects

Example:
```python
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12

def forward(self, ids: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
    ...
```

### Error Handling
- Use `assert` for internal invariants and debugging
- Raise exceptions with descriptive messages for user-facing errors
- Handle expected error cases gracefully (e.g., file not found, invalid input)

Example:
```python
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    # ... implementation
```

### Code Organization
- Keep related functionality together in modules
- Use `__init__.py` to mark packages
- Put entry points in `scripts/` directory
- Core logic in `nanochat/` package

### Documentation
- Use docstrings for classes and complex functions
- Keep module-level docstrings at the top of files
- Comment non-obvious code sections

### PyTorch Conventions
- Use `nn.Module` for neural network components
- Follow PyTorch naming: `forward`, `__init__`, `parameters()`
- Use `torch.nn.functional` for stateless operations (F.relu, F.layer_norm)
- Device placement: `model.to(device)`, `tensor.to(device)`

### Testing Guidelines
- Test files go in `tests/` directory
- Name test files: `test_*.py`
- Name test functions: `test_*`
- Use pytest markers for slow tests: `@pytest.mark.slow`
- Create mock classes for testing without full model loading

Example:
```python
import pytest

@pytest.mark.slow
def test_full_pipeline():
    """This test is slow and can be skipped with -m 'not slow'"""
    ...
```

### Git Practices
- Make focused, atomic commits
- Write descriptive commit messages
- Don't commit large binary files or secrets
- Use `.gitignore` to exclude build artifacts, cache, etc.

### Performance Considerations
- Use `@torch.no_grad()` for inference
- Use `torch.compile()` when beneficial
- Consider KV cache for inference
- Use mixed precision (FP8) where supported
- Profile with PyTorch profiler for bottlenecks

### Common Patterns
- DataParallel for multi-GPU inference
- torchrun for distributed training
- wandb for experiment tracking
- DistributedDataSampler for distributed data loading
