def Optimizer_input(context: str) -> str:
    return f"""# Code Optimization Task

## Experimental Context
{context}

## Objective
Generate an optimized version of the existing code to improve performance or resource usage without altering functionality.

### Steps
1. **Inspect** the current code using `read_code_file`.
2. **Identify** bottlenecks (e.g., shape mismatches, quadratic operations).
3. **Apply** targeted optimizations (chunking, dynamic shapes, einsum vs matmul).
4. **Implement** the optimized code with `write_code_file`, keeping `@torch.compile` decorators.

## Deliverables
- **name**: Identifier for the optimized variant.
- **motivation**: Clear reasoning for each optimization applied.
"""

def Optimizer_input_dedup(context: str, repeated_context: str) -> str:
    return f"""# Code Optimization Task (Deduplication Mode)

## Experimental Context
{context}

## Previously Attempted Optimizations
{repeated_context}

## Objective
Produce a new optimized code variant that avoids repeating past optimization strategies.

### Steps
1. **Review** past optimization motivations above.
2. **Inspect** code via `read_code_file`.
3. **Design** alternative optimizations not previously applied.
4. **Implement** them via `write_code_file`.
5. **Ensure** the motivation is novel compared to past attempts.

## Deliverables
- **name**: Unique optimized identifier.
- **motivation**: Explanation of new optimization approach and why it’s distinct.
"""
