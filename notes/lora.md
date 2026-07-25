# LoRA (Low-Rank Adaptation) — Notes

## The problem it solves
- Fine-tuning a model = adjusting the numbers in its weight grids (matrices) so it behaves better on a specific task.
- A "correction" (the change needed) is normally a full-size grid, same size as the original — expensive to compute, store, and duplicate for every fine-tuned version.

## The core idea
- Instead of learning a full-size correction grid directly, LoRA learns it as the **product of two much smaller, thin grids**.
- Example: to produce a 1000×1000 correction (1,000,000 numbers), use:
  - Grid A: 1000 rows × 8 columns (8,000 numbers)
  - Grid B: 8 rows × 1000 columns (8,000 numbers)
  - Multiply A × B → get a full 1000×1000 grid.
- Only 16,000 numbers are trained/stored instead of 1,000,000. Huge saving.

## "Rank"
- The shared small dimension (8 in the example above) is called the **rank**.
- Low rank = betting that the needed correction is "simple" underneath, so it can be compressed into a small number of dimensions without losing what matters.
- Typical values: 8, 16, 64.

## Putting it together
- Original model weights = **frozen**, never touched.
- LoRA adds a small trainable "patch" on top: `output = original_grid + (A × B)`.
- Only the patch (A and B) is trained and saved.
- At inference: `effective_grid = original_grid + (A × B)`.

## Why it's useful
- Cheap to train (few numbers to update).
- Cheap to store (tiny adapter files vs. a full model copy per task).
- Swappable: keep one frozen base model, swap in different small LoRA adapters for different tasks/domains.

## Connection to the RAG project
- vLLM has an `enable_lora` flag: lets the serving engine support LoRA adapters at request time.
- `max_loras`: how many adapters can be active/loaded concurrently.
- `lora_request`: lets a specific request choose which adapter to use.
- Relevant if the project ever needs to serve multiple fine-tuned variants of a model without duplicating the whole thing.
