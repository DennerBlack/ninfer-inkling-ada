# Inkling 2 Mini on the RTX 4060 Laptop GPU

Windows · 8 GB RTX 4060 Laptop GPU · Core i7-13700H · 32 GB RAM

One text request at a time. The model is Inkling 2 Mini UD-IQ4_XS, with routed experts in pinned host RAM and their hot weights in a GPU cache. The build and model artifact were held fixed throughout.

## 64K and 128K with Q4_0 KV

Each run asked the model to retrieve three exact record values placed near 10%, 50% and 94% of the input. The context window and KV capacity were set to the same value.

| Window | Chunk | Input tokens | Prefill | Generation | First token | Retrieval |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 65,536 | 1,536 | 61,265 | 539.8 tok/s | 15.9 tok/s | 1m 53.5s | 3/3 |
| 65,536 | 2,560 | 61,265 | 683.2 tok/s | 17.8 tok/s | 1m 29.7s | 3/3 |
| 131,072 | 1,536 | 124,065 | 379.7 tok/s | 12.0 tok/s | 5m 26.8s | 3/3 |
| 131,072 | 2,560 | 124,065 | 453.0 tok/s | 14.3 tok/s | 4m 34.0s | 3/3 |

The 2,560-token rows use tiled Q4_0 decode. All four requests returned the same three values in the same order. Total generation length, including reasoning, changed from 210 to 189 tokens at 64K and from 240 to 157 at 128K. The headroom figures below come from the engine capacity log. Pinned expert sources occupied 13,385 MiB in each run.

## Settings sweep

These are matched requests within each window. The short prompt had 36 input tokens and 114 output tokens. The other prompt had 4,407 input tokens and asked for one exact record value.

| Window | Chunk | Expert cache requested / actual (MiB) | Short generation (tok/s) | 4,407-token prefill (tok/s) | Generation after 4,407 tokens (tok/s) | 4,407-token request time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 512 | 1,536 / 1,535 | 18.6 | 252.6 | 17.5 | 22.44 s |
| 131,072 | 512 | 1,536 / 1,535 | 18.8 | 252.2 | 17.6 | 22.43 s |
| 65,536 | 512 | 2,304 / 2,301 | 21.2 | 272.1 | 19.6 | 20.65 s |
| 65,536 | 1,536 | 1,920 / 1,918 | 20.8 | 733.7 | 19.3 | 10.48 s |
| 131,072 | 512 | 1,792 / 1,792 | 19.5 | 259.2 | 18.2 | 21.81 s |
| 131,072 | 1,536 | 1,408 / 1,408 | 19.0 | 703.3 | 18.1 | 11.06 s |
| 65,536 | 1,536 | 2,048 / 2,044 | 21.6 | 741.6 | 19.7 | 10.32 s |

The 1,536-token chunks increased prefill by 2.94× at 64K and 2.79× at 128K on the matched 4,407-token input. The visible text matched across the sweep. The larger chunks changed the generation length for that input from 88 to 87 tokens.

The 128K profile with a 1,536-token chunk and a 1,536 MiB cache reached only 189 MiB of startup VRAM headroom. The benchmark used 1,408 MiB for that profile and recorded 313 MiB of headroom. The 64K profile used a 2,048 MiB cache target.


### Larger chunks and tiled decode

| Window | Chunk | Expert cache requested / actual (MiB) | Short generation (tok/s) | 4,407-token prefill (tok/s) | Generation after 4,407 tokens (tok/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 2,048 | 1,856 / 1,853 | 21.5 | 723.6 | 19.7 |
| 65,536 | 2,560 | 1,536 / 1,535 | 20.7 | 976.4 | 19.1 |
| 65,536 | 3,072 | 1,472 / 1,282 | 20.2 | 908.1 | 19.0 |
| 131,072 | 2,048 | 1,216 / 1,169 | 18.9 | 680.4 | 17.9 |
| 131,072 | 2,560 | 1,024 / 981 | 18.8 | 941.4 | 17.7 |
| 131,072 | 3,072 | 832 / 759 | 19.0 | 885.4 | 17.8 |

The 64K admission with a 2,560-token chunk and a 1,664 MiB cache had 193 MiB free at startup. The benchmark exited at its 200 MiB guard. A 1,536 MiB cache left 319 MiB free.

On a matched 15,617-token input in the 128K window, the 2,560-token chunk with tiled decode reduced total request time from 28.73 s to 22.44 s. The generated text, token count and reasoning hash matched.

| Chunk | Decode option | Prefill (tok/s) | Generation (tok/s) | Request time |
| ---: | :--- | ---: | ---: | ---: |
| 1,536 | Default | 659.4 | 17.3 | 28.73 s |
| 2,560 | Default | 892.0 | 17.2 | 22.60 s |
| 2,560 | Q4_0 tiled decode | 892.8 | 17.7 | 22.44 s |
| 2,560 | Cache bias by loop passes | 901.7 | 16.6 | 22.58 s |

<details>
<summary>Initial 8K screen</summary>

| GPU expert cache target (MiB) | DSpark drafts | GPU share of expert misses | Short generation (tok/s) | Accepted drafts |
| ---: | :--- | :--- | ---: | ---: |
| 1536 | Off | 1 | 18.8 | — |
| 2048 | Off | 1 | 20.5 | — |
| 2304 | Off | 1 | 21.5 | — |
| 1536 | Off | Auto | 17.9 | — |
| 1536 | 1 | 1 | 19.4 | 51/62 |
| 1536 | 2 adaptive | 1 | 18.7 | 66/92 |
| 1536 | 3 adaptive | 1 | 18.8 | 75/107 |
| 1536 | 4 adaptive | 1 | 18.4 | 74/114 |
| 1536 | Off | 1 | 18.7 | — |

The GPU expert cache was the larger gain for this prompt. The 1,536 MiB and 2,048 MiB configurations also produced identical visible responses for the JSON, code and retrieval cases. Both prose requests hit the same 384-token output limit. Both prefix-repeat requests recorded zero cache hits.

</details>

## Reproduce

The serving binary identified itself as `2026-11-21.4+bd8195f42e87-hel-q4kv` with CUDA 13.3.33, SM89. The locally packed artifact SHA-256 was `cdaa3d20474fb0874a4f899d29cd04e43993e32e16736d558d3fd3f71132d924`. The calibrated device profile supplied 120 routed keys.

Run the server with one profile at a time. The Windows expert-cache ceiling stays enabled so actual cache residency can follow the local WDDM budget.

The larger chunk improves long requests. The 1,536-token chunk gave slightly higher generation rates on the short prompt, 21.6 versus 21.2 tok/s at 64K and 19.0 versus 18.8 at 128K.

```powershell
$weights = 'C:\models\Inkling-2-Mini-UD-IQ4_XS.ninfer'
$window = 65536
$cache = 1600
# For 128K, set $window to 131072 and $cache to 1024.

$flags = @(
    'ATTENTION_GRAPHS', 'FUSED_EXPERTS', 'EXPERT_GRAPHS', 'LEND_WORKSPACE',
    'PROMOTE_LOCAL', 'COPY_EVENT', 'SPARSE_STAGES', 'REUSE_ROUTE_BOUNDS'
)
foreach ($flag in $flags) {
    [Environment]::SetEnvironmentVariable("HEL_INKLING_$flag", "1", "Process")
}
$env:HEL_INKLING_PROMPT_MMA = '3'
$env:HEL_INKLING_EXPERT_CACHE_CEILING = '1'
$env:HEL_INKLING_DECODE_TILE = '1'

& .\ninfer-serve.exe $weights --model-id inkling-2-mini --host 127.0.0.1 --port 18771 `
    --no-webui --max-context $window --kv-capacity $window --max-concurrency 1 --kv-dtype q4_0 `
    --expert-residency host --embedding-residency host --expert-cache-mib $cache `
    --expert-dma-share 1 --expert-cpu-threads 2 --prefill-chunk 2560 --host-cache-mib 0 `
    --no-thinking --spec none --device-profile auto --device-profile-path .\device-profiles.json
```

The benchmark sends a single user message to `/v1/chat/completions` at temperature zero. It uses 384 output tokens for short and 4,407-token requests and 768 for near-limit requests. All timed rows above recorded zero cached input tokens.

<details>
<summary>Near-limit retrieval fixture</summary>

```powershell
$count = if ($window -eq 65536) { 3110 } else { 6250 }
$first = [int]($count / 10)
$middle = [int]($count / 2)
$last = [int]($count * 0.94)
$records = for ($record = 1; $record -le $count; $record++) {
    $tag = switch ($record) {
        $first  { 'ORION-P9A2' }
        $middle { 'KESTREL-Q7F4' }
        $last   { 'LYNX-T5M8' }
        default { 'pending' }
    }
    "Record $record - manifest status $tag; storage bin b$($record % 19); item total $($record % 17)."
}
$question = "Find the manifest status for records $first, $middle and $last. Reply with those three statuses in order, separated by commas."
$prompt = (@($records) + $question) -join [char]10
$body = @{
    model = 'inkling-2-mini'
    messages = @(@{ role = 'user'; content = $prompt })
    temperature = 0
    max_tokens = 768
} | ConvertTo-Json -Depth 4 -Compress
Invoke-RestMethod http://127.0.0.1:18771/v1/chat/completions -Method Post -ContentType application/json -Body $body
```

</details>

The measured scenarios cover cold prefill and generation with one text request at a time.
