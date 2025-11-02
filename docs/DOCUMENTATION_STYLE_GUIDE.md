# Lumiére Documentation Style Guide

This guide defines the documentation standards for the Lumiére project. It ensures consistency, clarity, and maintainability across the codebase.

**Companion file**: See `DOCUMENTATION_DEFINITIONS.md` for standard terminology and definitions.

## Target Audience

Documentation assumes readers have:
- Working knowledge of Python and PyTorch
- Understanding of transformer architecture fundamentals
- Familiarity with neural network concepts (attention, embeddings, feed-forward networks)

## General Principles

1. **Be concise but complete** - Include all necessary information without verbosity
2. **Be consistent** - Use the same terminology and phrasing patterns throughout
3. **Be precise** - Use technical terms correctly and consistently
4. **Avoid redundancy** - Don't repeat information unless context requires it
5. **Use imperative mood** - Start docstrings with verbs: "Initialize", "Perform", "Compute", "Pass"

## Docstring Structure

### Class Docstrings

Use a three-part structure:

1. **Class-level docstring**: Brief description of what the class is
2. **`__init__` docstring**: Detailed initialization documentation with Args
3. **`forward` docstring**: Detailed forward pass documentation with Args and Returns

**Example:**
```python
class TransformerBlock(nn.Module):
    """A decoder transformer block."""

    def __init__(self, embedding_size: int, num_heads: int):
        """Initialize a transformer block.

        Args:
            embedding_size: The dimensionality of the token embeddings.
            num_heads: The number of attention heads.

        """
        super().__init__()
        # implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a batch of token embeddings through the transformer block.

        Args:
            x: A batch of token embeddings of shape
                `(batch_size, context_size, embedding_size)`.

        Returns:
            A batch of output embeddings of shape
            `(batch_size, context_size, embedding_size)`.

        """
        # implementation
```

**Class docstrings with additional context:**

When a class implements a specific technique from a paper or requires additional context, add a second paragraph to the class docstring:

```python
class LinearFeedForward(nn.Module):
    """A position-wise feed-forward network.

    This class implements the position-wise feed-forward network as described in the
    paper `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    """
```

### Function Docstrings

Private helper functions should have:
1. One-line summary (imperative mood)
2. Args section (if parameters need explanation)
3. Returns section

**Example:**
```python
def _create_causal_mask(context_size: int, padding_mask: torch.Tensor = None) -> torch.Tensor:
    """Create a causal mask for the attention operation.

    This mask is used to prevent each token from attending to itself, future tokens,
    and padding tokens (if a padding mask is provided).

    Args:
        context_size: The maximum number of tokens in a sequence.
        padding_mask: A boolean mask indicating which of the tokens are padding
            tokens, with `True` indicating a padding token and `False` for
            non-padding tokens.

    Returns:
        A mask of shape `(1, 1, context_size, context_size)`.
    """
```

### Property Docstrings

Properties should have minimal, single-line docstrings:

```python
@property
def vocab_size(self) -> int:
    """The number of unique tokens in the vocabulary."""
    return self._vocab_size
```

**Properties that return neural network layers:**

For properties that expose internal layers, describe what the layer does functionally:

```python
@property
def up_proj(self):
    """The linear layer that projects embeddings to the hidden dimension."""
    return self._layers.up_proj

@property
def down_proj(self):
    """The linear layer that projects hidden states back to embeddings."""
    return self._layers.down_proj
```

## Formatting Rules

### 1. Article Usage

- Class docstrings: Use "A" (not "The")
  - ✅ "A decoder transformer block."
  - ❌ "The decoder transformer block."

### 2. Verb Mood

All docstrings use **imperative mood**:
- ✅ "Initialize", "Perform", "Pass", "Create", "Compute", "Apply", "Split", "Concatenate"
- ❌ "Initializes", "Performs", "Passes", "Creates", "Computes", "Applies", "Splits", "Concatenates"

### 3. Args Section

- Start each arg description with "The" (for clarity)
- End with a period
- Include shape information inline with backticks when describing tensors
- **Use exact definitions from `DOCUMENTATION_DEFINITIONS.md`**

```python
Args:
    vocab_size: The number of unique tokens in the vocabulary.
    x: A batch of token embeddings of shape
        `(batch_size, context_size, embedding_size)`.
```

### 4. Returns Section

- Start with "A" for single returns
- For tuples, use: "A tuple of [X] and [Y]. The [X] have shape `...` and the [Y] have shape `...`"
- Avoid markdown lists (doesn't render well in pydoc)
- Remove words like "respectively", "containing" for conciseness
- **Use exact phrasing from `DOCUMENTATION_DEFINITIONS.md` when available**

```python
Returns:
    A tuple of output embeddings and attention weights. The output embeddings
    have shape `(batch_size, context_size, embedding_size)` and the attention
    weights have shape `(batch_size, num_heads, context_size, context_size)`.
```

### 5. Shape Specifications

- **Input shapes**: Include in Args section
- **Output shapes**: Include in Returns section
- Always use backticks: `` `(batch_size, context_size, embedding_size)` ``
- Use "shape" not "the shape" for brevity
- Break long shapes to next line for readability
- Follow naming conventions from `DOCUMENTATION_DEFINITIONS.md`

### 6. Blank Lines

- Add blank line before closing `"""` in multi-line docstrings
- No blank line for single-line docstrings

```python
"""Single line docstring."""

"""Multi-line docstring.

Args:
    param: Description.

Returns:
    Description.

"""
```

## Standard Terminology

**IMPORTANT**: Always consult `DOCUMENTATION_DEFINITIONS.md` for canonical definitions of:
- Core parameters (`vocab_size`, `embedding_size`, `context_size`, etc.)
- Common input/output descriptions
- Tensor shape specifications
- Standard phrasing patterns

Do not paraphrase or modify these definitions - use them exactly as written.

## Common Patterns

### Describing Tensors

```python
x: A batch of token embeddings of shape
    `(batch_size, context_size, embedding_size)`.
```

### Describing Optional Parameters

```python
padding_mask: A boolean mask indicating which of the tokens in the batch
    are padding tokens, with `True` indicating the presence of a padding
    token and `False` for non-padding tokens. Expected to have the shape:
    `(batch_size, context_size)`.
```

### Describing Tuple Returns

```python
Returns:
    A tuple of [first_element] and [second_element]. The [first_element]
    have shape `(...)` and the [second_element] have shape `(...)`.
```

## What NOT to Do

❌ **Don't use markdown lists in docstrings** (poor pydoc rendering)
```python
# BAD
Returns:
    A tuple containing:
        - Output embeddings of shape `(...)`
        - Attention weights of shape `(...)`
```

❌ **Don't use "respectively" unnecessarily**
```python
# BAD
Returns:
    A tuple of tensors containing the output embeddings and attention weights,
    respectively.
```

❌ **Don't include type hints in Args** (already in signature)
```python
# BAD
Args:
    num_heads (int): The number of attention heads.

# GOOD
Args:
    num_heads: The number of attention heads.
```

❌ **Don't repeat "the" before "shape"**
```python
# BAD: have the shape
# GOOD: have shape
```

❌ **Don't mix tenses**
```python
# BAD: "Creates a mask..." (present tense)
# GOOD: "Create a mask..." (imperative)
```

❌ **Don't paraphrase standard definitions**
```python
# BAD: vocab_size: Number of tokens in vocabulary
# GOOD: vocab_size: The number of unique tokens in the vocabulary.
# (exact match from DOCUMENTATION_DEFINITIONS.md)
```

## Workflow

### For Humans

1. **Before documenting**:
   - Read this style guide
   - Check `DOCUMENTATION_DEFINITIONS.md` for standard definitions
   - Look at similar code in the codebase for patterns

2. **While documenting**:
   - Use exact definitions from `DOCUMENTATION_DEFINITIONS.md`
   - Follow the formatting rules above
   - Be consistent with existing documentation

3. **After documenting**:
   - Run `ruff check <file>` to verify linting
   - Run `ruff format --check <file>` to verify formatting
   - Review the checklist below

### For AI Assistants

1. **Read both files first**: This style guide + `DOCUMENTATION_DEFINITIONS.md`
2. **Match existing patterns**: If similar code exists, follow its documentation style exactly
3. **Use exact definitions**: Copy from `DOCUMENTATION_DEFINITIONS.md`, don't paraphrase
4. **Preserve formatting**: Maintain blank lines, indentation, and structure
5. **Verify with tools**: Always run `ruff check` and `ruff format --check` after changes
6. **Context matters**: Some repetition is necessary (e.g., padding_mask in each method)

#### Common AI Tasks

**Task: Document a new class**
1. Add brief class docstring (one line, imperative, starts with "A")
2. Document `__init__` with all Args (use definitions from `DOCUMENTATION_DEFINITIONS.md`)
3. Document `forward` with Args and Returns including shapes
4. Add property docstrings if needed

**Task: Update parameter descriptions**
1. Check `DOCUMENTATION_DEFINITIONS.md` first
2. If it's a standard parameter, use the exact wording
3. If it's new, add it to `DOCUMENTATION_DEFINITIONS.md` following existing patterns

**Task: Fix inconsistencies**
1. Identify the correct definition from `DOCUMENTATION_DEFINITIONS.md`
2. Search for all occurrences of the term being changed
3. Update all occurrences to match the standard
4. Verify with grep/search that all instances are consistent

**Task: Add new standard definitions**
1. Add to `DOCUMENTATION_DEFINITIONS.md` following the template
2. Use it consistently across the codebase
3. Verify the definition works in multiple contexts

## Validation Checklist

Before committing documentation changes:

- [ ] All docstrings use imperative mood
- [ ] Class docstrings use "A" not "The"
- [ ] Args descriptions start with "The" and end with period
- [ ] Standard definitions match `DOCUMENTATION_DEFINITIONS.md` exactly
- [ ] Shape info in Args (inputs) and Returns (outputs)
- [ ] Backticks around all code elements (shapes, variables, `True`, `False`)
- [ ] No markdown lists in docstrings
- [ ] Blank line before closing `"""` in multi-line docstrings
- [ ] `ruff check` passes
- [ ] `ruff format --check` passes

## Reference Implementation

See `lumiere/releases/Lumiere1M/model.py` for a complete reference implementation of all these patterns.

## Questions or Clarifications

When in doubt:
1. Check `DOCUMENTATION_DEFINITIONS.md` for standard definitions
2. Look for similar patterns in `lumiere/releases/Lumiere1M/model.py`
3. Follow the "What NOT to Do" section to avoid common mistakes
4. Prioritize consistency over cleverness
