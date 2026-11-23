# Inkling 2 Mini on the RTX 5070 Laptop GPU

Windows · RTX 5070 Laptop GPU · 8 GB VRAM

The benchmark ran Inkling 2 Mini UD-IQ4_XS with Q4_0 KV and staged host experts. At each KV capacity, the request asked for three record values placed near 10%, 50%, and 94% of the input. The [retrieval fixture](inkling-ada-laptop.md#reproduce) shows how the prompt is constructed.

## Long-context results

| KV capacity, tokens | Prefill chunk, tokens | Input tokens | Prefill, tok/s | Generation, tok/s | TTFT, s | Records retrieved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 1,536 | — | 655.6 | 19.9 | — | 3/3 |
| 65,536 | 2,560 | 61,265 | 903.1 | 20.1 | 67.9 | 3/3 |
| 131,072 | 1,536 | — | 488.7 | 15.4 | — | 3/3 |
| 131,072 | 2,560 | 124,065 | 613.0 | 16.4 | 202.4 | 3/3 |

Both 2,560-token runs returned the three values in order. Input counts and time to first token are available for those two runs. The exact revision of the local build and the model artifact hash are unavailable for these measurements.
