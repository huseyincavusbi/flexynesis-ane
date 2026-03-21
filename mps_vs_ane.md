# MPS vs ANE Comparison
flexynesis 1.1.7, Dataset1, Apple M4, macOS 26.3

## Timing Metrics Explained
- **real**: Wall-clock time (actual elapsed time from start to finish)
- **sys**: System time (OS kernel operations like memory management, I/O)

## Training Time (seconds)

| Model | MPS real | MPS sys | ANE real | ANE sys |
|-------|----------|---------|----------|---------|
| DirectPred | 60.47 | 4.93 | 26.18 | 0.90 |
| Supervised VAE | 381.82 | 27.46 | 204.34 | 5.52 |
| GNN | 84.95 | 4.56 | 53.83 | 1.19 |
| CrossModalPred | 338.36 | 24.12 | 219.95 | 5.71 |

**ANE is 1.5–2.3x faster overall; system overhead reduced 3.8–5.5x, indicating more efficient kernel dispatching.**

## Validation Performance

| Model | Metric | MPS | ANE |
|-------|--------|-----|-----|
| **DirectPred** | Erlotinib Loss | 0.00415 | 0.00609 |
| | Val Speed (it/s) | 126.45 | 262.93 |
| **Supervised VAE** | Erlotinib Loss | 0.00549 | 0.00435 |
| | Latent Dist (MMD) | 0.73935 | 0.73483 |
| | Val Speed (it/s) | 77.46 | 114.95 |
| **GNN** | Erlotinib Loss | 0.00430 | 0.00441 |
| | Val Speed (it/s) | 80.80 | 119.16 |
| **CrossModalPred** | Erlotinib Loss | 0.00525 | 0.00575 |
| | Latent Dist (MMD) | 0.71668 | 0.70274 |
| | Val Speed (it/s) | 76.93 | 113.12 |

**ANE validation is 1.5–2.1x faster with minimal/no accuracy loss.**