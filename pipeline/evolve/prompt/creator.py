def Creator_input(context: str) -> str:
    return f"""# Code Creation Task

## Experimental Context
{context}

## Objective
Generate a brand‑new code variant that advances the architecture or algorithmic design.

### Steps
1. **Inspect** the existing implementation using `read_code_file`.
2. **Design** novel modifications that preserve sub‑quadratic complexity.
3. **Implement** the complete code via `write_code_file`, including sensible defaults and `**kwargs`.
4. **Ensure** `@torch.compile` is applied to performance‑critical functions.

## Deliverables
- **name**: A concise identifier (no timestamp).
- **motivation**: Clear explanation of what was changed and why.
"""

def Creator_input_dedup(context: str, repeated_context: str) -> str:
    return f"""# Code Creation Task (Deduplication Mode)

## Experimental Context
{context}

## Previously Attempted Motivations
{repeated_context}

## Objective
Produce a new code variant with a unique motivation that does _not_ repeat past designs.

### Steps
1. **Review** past motivations above.
2. **Inspect** the code with `read_code_file`.
3. **Innovate** in a direction orthogonal to those repeated patterns.
4. **Implement** via `write_code_file`, preserving complexity and interfaces.
5. **Validate** that your motivation is distinct before proceeding.

## Deliverables
- **name**: Unique identifier.
- **motivation**: Explanation highlighting how this differs from previous attempts.
"""
