# grand-pattern-mono

**The corrected Grand Pattern — vibe is mono-dimensional, JEPA is a weighted reading.**

## Core Insight

Vibe is ONE number (f64), not a vector. This makes conservation trivially hold:
- Total vibe = sum of all room vibes
- Adding/removing rooms adjusts the total directly
- No GC drift, no magnitude mismatch, no conservation violations

## Architecture

- **Vibe = f64** — One number per room. Conservation is arithmetic.
- **JEPA** — Weighted moving average predictor. Weights learn by inverse-error reinforcement: readings close to actual get boosted, far readings get dampened. Each room develops its own weights based on its own history.
- **Murmur** — Gossip message carrying one number + surprise + TTL.
- **CellGraph** — Rooms connected by edges, with diffusion (gossip) and tick-driven updates.

## Key Properties (verified by 27 tests)

1. **Conservation holds trivially** — vibe is a scalar, total is tracked directly
2. **JEPA weights learn** — rooms with different histories develop different predictions
3. **Diffusion converges** — connected rooms equalize their vibes through gossip
4. **Surprise cascades** propagate through graph edges with decay
5. **TTL-based decay** — murmurs expire, preventing stale gossip

## Running

```bash
cargo test    # 27 tests
```

This is a library crate — no binary. Use it as a dependency.

## Testing

```
running 27 tests — all pass
```

Covers: conservation, JEPA learning, diffusion convergence, surprise cascade, TTL decay, edge cases (empty graph, single room, oscillating input).

## License

MIT
