# Circle Packing Reproduction

This directory contains a simplified reproduction of the circle packing experiment from the paper "optimize_anything: A Universal API for Optimizing any Text Parameter" (arXiv:2605.19633v1).

## Paper Context

The paper demonstrates that LLM-based optimization can evolve a simple greedy circle packing algorithm into a sophisticated 900+ line bilevel optimizer, achieving a score of 2.63598 (sum of radii for n=26 circles in a unit square).

## This Reproduction

This reproduction implements the **baseline seed algorithm** described in the paper:
- Concentric ring pattern: 1 center circle + 8 inner ring + 17 outer ring
- Greedy radius computation with constraint satisfaction
- Iterative refinement to maximize radii while avoiding overlaps

**This is a qualitative reproduction** showing the starting point of the optimization process, not the final evolved solution.

## Files

- `paper.txt` - Transcription of relevant sections from the paper
- `config/config.yaml` - Hydra root configuration
- `config/run/reproduction.yaml` - Experiment parameters with source annotations
- `src/main.py` - Circle packing implementation
- `requirements.txt` - Python dependencies

## Expected Results

The baseline algorithm produces a sum of radii around 1.8-2.0, compared to:
- Paper's seed baseline: ~2.0
- Paper's optimized solution: 2.63598

This demonstrates the ~32% improvement potential through LLM-driven optimization.

## Validation

All files have been syntax-checked:
- Python: ✓ Valid AST
- YAML: ✓ Valid configuration

## Run Instructions (for workflow)

```bash
uv run python -u -m src.main run=reproduction results_dir=results
```

## Outputs

When run, produces:
- `results/repro.png` - Visualization of circle packing
- `results/result.json` - Metrics and summary in required schema
