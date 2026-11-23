<h1 align="center">NInfer · Inkling Ada</h1>

<p align="center">
  Inkling 2 Mini with a 256K context on a single 16 GB RTX 4080 SUPER.
</p>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> · <a href="#quick-start">Quick start</a> · <a href="#implementation">Implementation</a>
</p>

## Benchmarks

Windows · RTX 4080 SUPER 16 GB · Core i7-12700F · 64 GB RAM

| Input tokens | Prefill | Generation | First token |
| ---: | ---: | ---: | ---: |
| 9,700 | 2,140 tok/s | **79.2 tok/s** | 4.6 s |
| 56,456 | 2,110 tok/s | **65.3 tok/s** | 26.8 s |

<details>
<summary>Test settings</summary>

One slot with a GPU KV cache sized for 262,144 tokens. The timed text requests had zero prefix-cache hits and generated 320 tokens each. Prefill chunks were 3,072 tokens. DSpark used two draft tokens and n-gram lookup.

The Q4_0 GPU KV allocation was 3,035 MiB. The GPU expert cache held 5,662 MiB, and pinned host expert weights took about 13.1 GiB. The timed runs used CUDA 13.3 and a locally packed UD-IQ4_XS artifact.

</details>

### RTX 4060 Laptop GPU · 8 GB

One text request at a time with Q4_0 KV, staged host experts, and a 2,560-token prefill chunk.

| Context window | Input tokens | Prefill | Generation | First token | Startup VRAM free |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 61,265 | 683.2 tok/s | **17.8 tok/s** | 1m 29.7s | 255 MiB |
| 131,072 | 124,065 | 453.0 tok/s | **14.3 tok/s** | 4m 34.0s | 297 MiB |

[Laptop benchmark and launch settings](docs/inkling-ada-laptop.md)

## Quick start

### 1. Build the server

Open an x64 Developer PowerShell with MSVC, CUDA, CMake, Ninja, and vcpkg. Put CUDA on `PATH` and point `VCPKG_ROOT` at your vcpkg installation. [Windows build guide](docs/build-windows.md)

```powershell
cmake -S . -B build-ada -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-ada --target ninfer-serve
```

### 2. Download the model

The WaveCut package includes UD-IQ4_XS text weights, Q8_0 vision, and a DSpark drafter. The download is 16.5 GB. [Model card](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3)

```powershell
hf download WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3 `
  Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer `
  --revision 50d6ad60cfcd170633ead778e3a09958fc1e7887 --local-dir models
```

### 3. Start a local server

The Ada profile uses one slot and runs KV, vision, and DSpark on the GPU. Routed experts live in pinned host RAM with a GPU cache. Lower `--expert-cache-mib` if the GPU has less memory available at startup.

```powershell
$inklingFlags = @(
  'ATTENTION_GRAPHS', 'FUSED_EXPERTS', 'EXPERT_GRAPHS', 'LEND_WORKSPACE',
  'VERIFY_PAIRS', 'ROUTE_GRAPHS', 'PROMOTE_LOCAL', 'COPY_EVENT',
  'SPARSE_STAGES', 'REUSE_ROUTE_BOUNDS'
)
foreach ($flag in $inklingFlags) {
  [Environment]::SetEnvironmentVariable("HEL_INKLING_$flag", '1', 'Process')
}
$env:HEL_INKLING_PROMPT_MMA = '3'
$env:HEL_INKLING_EXPERT_CACHE_CEILING = '1'
$env:HEL_INKLING_ALLOCATION_RESERVE_MIB = '300'

.\build-ada\apps\ninfer-serve.exe .\models\Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer `
  --model-id inkling-2-mini --host 127.0.0.1 --port 8080 --no-webui `
  --max-context 262144 --kv-capacity 262144 --max-concurrency 1 --kv-dtype q4_0 `
  --expert-residency host --embedding-residency host --expert-cache-mib 5663 `
  --expert-dma-share 1 --prefill-chunk 3072 --vision `
  --spec dspark --draft-tokens 2 --draft-min-p 0.5 --adaptive-mtp --lookup-ngram 3 `
  --host-cache-mib auto --host-cache-max-mib 7168 --host-cache-reserve-mib 2048
```

Send a request from a second terminal.

```powershell
$request = @{
  model = 'inkling-2-mini'
  messages = @(@{ role = 'user'; content = 'Describe Saturn in one sentence.' })
  max_tokens = 128
} | ConvertTo-Json -Depth 4
Invoke-RestMethod 'http://127.0.0.1:8080/v1/chat/completions' `
  -Method Post -ContentType 'application/json' -Body $request
```

## Implementation

- Q4_0 KV storage for full attention in both Inkling and its drafter. Sliding-window attention uses F16.
- Ada CUDA kernels and graph capture for attention and expert routing, with asynchronous transfers between host RAM and the GPU expert cache.
- Smaller host snapshots for edited conversations. In a four-request A/B test, the compact path saved 208 MiB of host cache and produced identical text streams.

## Credits

Engine based on [NInfer-all 2026-11-21.4](https://github.com/iamwavecut/ninfer-all/releases/tag/2026-11-21.4) · [Upstream documentation](README.upstream.md) · [Apache-2.0](LICENSE)

Inkling 2 Mini by Thinking Machines Lab · [Quantization by Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) · [NInfer package by WaveCut](https://huggingface.co/WaveCut/Inkling-2-Mini-UD-IQ4_XS-NInfer-v3)
