# Inkling 2 Mini benchmark notes

## Desktop reference profile

Windows, RTX 4080 SUPER 16 GB, Core i7-12700F, 64 GB RAM. The reference is the original November 22, 2026 run of Inkling Ada with CUDA 13.3. Both requests used the same loaded model and reached the 320-token output limit at temperature zero, with zero prefix-cache hits. The November 23 rerun hit the Windows commit guard during startup, so the comparison uses the original completed run.

- One active slot and one 262,144-token GPU KV pool. Full-attention keys and values use Q4_0. Sliding-window state uses FP16. The runtime reported 3,035 MiB for KV.
- Prefill chunks of 3,072 tokens. The Q8_0 DSpark drafter used up to two tokens per round, a probability floor of 0.5 and adaptive width. A three-token n-gram lookup also proposed up to two tokens.
- All 13,385 MiB of routed expert sources were pinned in host RAM. The requested GPU expert cache was 5,663 MiB and the effective cache was 5,662 MiB.
- Text-only requests through the streaming Anthropic endpoint, with thinking disabled. The prompts used 475 and 2,700 synthetic archive records, followed by the same request to write a guide. The rendered prompts were 9,700 and 56,456 tokens.
- Prefill and generation are rates from the engine log. The comparison uses the original client's time to first streamed text. Each input length has one timed sample.

## Desktop engine comparison

The launch checks requested one slot, 262,144 tokens of GPU KV, Q4_0 keys and values, and 3,072-token prefill chunks. The timed cases use the same two rendered prompts, greedy sampling, a 320-token output budget and zero prefix-cache hits. Both timed configurations use DSpark with up to two proposals and three-token n-gram lookup.

| Build | Observed startup |
| :--- | :--- |
| Inkling Ada | 262,144-token GPU KV pool with Q4_0 full attention, FP16 sliding-window attention and DSpark. The original November 22 run completed both requests. |
| NInfer-all 2026-11-21.4 and 2026-11-23.7 | Both Windows binaries returned <code>invalid kv-dtype: q4_0</code>. |
| Strata 0.2.20 / Inkling 0.1.66 | The Inkling executable reported <code>kv_dtype=fp8</code>, <code>READY 37184</code> and <code>drafting=off</code>. |
| llama.cpp b12651 | The release allocated 262,144 Q4_0 full-attention cells on CUDA. The run hit the harness's Windows commit guard during compute graph reservation, before the server accepted requests. |
| llama.cpp banded test build | The local build based on <code>7db4a7064aff</code> reported one 262,144-token slot, Q4_0 full and sliding-window KV on CUDA, a 3,072-token microbatch, DSpark and n-gram lookup. |

Strata reduced the requested window and reported 188 MiB of free VRAM against the 205 MiB drafter requirement. The released llama.cpp build allocated 2,304 MiB for full-attention KV and 180 MiB for sliding-window KV at the requested chunk size. A diagnostic with a 256-token microbatch reached a 19,433 MiB CUDA compute buffer and hit the same Windows commit guard.

[Base NInfer-all release](https://github.com/iamwavecut/ninfer-all/releases/tag/2026-11-21.4) · [Current NInfer-all test release](https://github.com/iamwavecut/ninfer-all/releases/tag/2026-11-23.7) · [Strata test release](https://github.com/niko1221/strata/releases/tag/v0.2.20) · [llama.cpp test release](https://github.com/ggml-org/llama.cpp/releases/tag/b12651) · [Banded attention commit](https://github.com/unslothai/llama.cpp/commit/7db4a7064afffe7fbed81227473182559d227222)

### Timed configurations

| Engine | KV capacity, tokens | Input tokens | Prefill, tok/s | Generation, tok/s | TTFT, s |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Inkling Ada | 262,144 | 9,700 | 2,140.0 | 79.2 | 4.6 |
| llama.cpp banded | 262,144 | 9,700 | 438.2 | 13.6 | 22.4 |
| Inkling Ada | 262,144 | 56,456 | 2,110.0 | 65.3 | 26.9 |
| llama.cpp banded | 262,144 | 56,456 | 394.9 | 11.4 | 143.4 |

Each row is one completed request. Prefill and generation rates come from the respective engines. TTFT is the time from sending the request to the first streamed text. Inkling Ada used the Anthropic endpoint on November 22. The banded build used the rendered prompt IDs via the llama.cpp completion endpoint on November 23.

| Runtime setting | Inkling Ada | llama.cpp banded |
| :--- | :--- | :--- |
| Full-attention KV | GPU Q4_0 · 262,144 tokens | GPU Q4_0 · 262,144 tokens |
| Sliding-window KV | FP16 | Q4_0 |
| GPU expert cache | 5,662 MiB | 3,062.5 MiB |
| Expert sources | Pinned host RAM | Memory-mapped host weights |
| DSpark | Q8_0 · two proposals · adaptive width · 0.5 probability floor | Q8_0 · two proposals · 0.0 confidence floor |

The banded build accepted 163 and 162 DSpark tokens on the two inputs. Its Inkling drafter requires a zero confidence floor because this head is absent from the Q8_0 file. The timings describe the measured configurations, including their cache and drafting differences.

The banded build used the Unsloth commit above with local Inkling and DSpark changes. The exact binary requires that patch (SHA-256 `741fc8289c8b20a464fe1954a291e6d4354729154f6c9c81a0fdba42c17d4f0f`). The tested Windows server binary SHA-256 is `25643d1ca90de5aeb486d9d8d339750c5e18023d8342a91727001eb514c4d863`. A run where the server continued after DSpark initialization failed was excluded.

<details>
<summary>Banded build expert-cache trials</summary>

The two larger expert-cache targets finished the 9,700-token request at 19.9 tok/s (5,120 MiB requested) and 17.8 tok/s (4,096 MiB requested). The memory guard stopped both during the 56,456-token trial after free VRAM fell below the 280 MiB threshold used for those trials. The table uses the 3,072 MiB target, which completed both requests with the same 256K context, KV precision and prefill chunk.

</details>

### Reproduce the desktop workload

Confirm the allocated window and KV formats in the startup log, then send the same two prompts to a single slot. Set temperature to zero and the output limit to 320. Record engine-reported prefill and decode rates and time to the first streamed text. Verify the input and output counts and zero prefix-cache hits for each timed request.

<details>
<summary>Desktop prompt generator</summary>

~~~python
def prompt(label, rows):
    samples = (
        "The workshop inventory contains pressure gauges, a box of seals and calibrated reference weights.",
        "The observatory records hourly cloud cover, air temperature, wind speed and instrument alignment.",
        "The maintenance log includes inspection dates, replacement parts and discrepancies in serial numbers.",
        "The survey team recorded water samples upstream and downstream of the bridge at equal intervals.",
    )
    records = "\n".join(f"{i + 1}. {samples[i % len(samples)]}" for i in range(rows))
    return (
        f"This is test corpus {label}. Below are examples of recurring technical records.\n{records}\n"
        "Write a detailed practical guide for a new technician responsible for validating a large digital archive. "
        "Include a clear procedure, handling duplicates and inconsistent measurements, how to document corrections, "
        "and six examples of concrete checks. Write at least 600 words."
    )

cases = [prompt("Copper", 475), prompt("Quasar", 2700)]
~~~

Render each prompt as a single user message with the Inkling chat template, with thinking disabled and the generation prompt enabled. Expected lengths are 9,700 and 56,456 tokens. For the llama.cpp test, send the rendered token IDs to <code>/completion</code> with <code>cache_prompt=false</code>.

The SHA-256 of the 9,700-token ID sequence is <code>cc6aac5c813f7bad036871beadd57115716cadadfb45ab2fb546a499cb54f6cb</code>. The 56,456-token sequence is <code>e557c252ed6ba59daa30a2a048e0e489993001c20895fb2d5abb9a02c7c519b6</code>. Each ID is encoded as a little-endian unsigned 32-bit integer.

The banded run used `chat_template.jinja` (SHA-256 `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc`) and `tokenizer.hf.json` (SHA-256 `09c0013e8a235bf5974bddea8a62c39b72b61da6d38c850bbaca16090c4f4241`) from the Strata 0.2.19 Inkling tokenizer bundle.

</details>

<details>
<summary>Streaming request settings</summary>

Send each Inkling Ada request to `/v1/messages` with a fresh conversation ID. In the body below, `text_prompt` is the generated text.

~~~python
{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": text_prompt}],
    "thinking": {"type": "disabled"},
    "temperature": 0,
    "max_tokens": 320,
    "stream": True,
}
~~~

For the llama.cpp banded build, the `/completion` body is below. Here `prompt_ids` is the rendered and tokenized prompt. Both Inkling Ada requests also reached their output limit.

~~~python
{
    "prompt": prompt_ids,
    "n_predict": 320,
    "temperature": 0,
    "ignore_eos": True,
    "cache_prompt": False,
    "stream": True,
}
~~~

The banded build ran a 24-token warm-up before timing the two requests. The observed startup log must show one 262,144-token slot, Q4_0 on CUDA and successful speculative context initialization. Check DSpark acceptance in the request logs.

</details>

<details>
<summary>Windows launch arguments</summary>

Inkling Ada uses the [README launch recipe](../README.md#3-start-a-local-server). The measured build identifies itself as <code>2026-11-21.4+bd8195f42e87-hel-q4kv</code>.

The released NInfer binaries were given <code>--max-context 262144 --kv-capacity 262144 --max-concurrency 1 --kv-dtype q4_0</code> with the same NInfer artifact.

Strata Inkling was started with the following arguments.

~~~powershell
.\strata-inkling.exe --serve --native $weights --max-context 262144 --parallel 1 `
  --kv q4_0 --kv-offload off --draft dspark --dspark $draft --chunk 3072 `
  --vram-reserve-mib 300 --ram-experts-gb 13.2 --cpu-threads 4 --vram-experts-gb 5.5
~~~

llama.cpp b12651 was checked with the mainline-layout weights and these arguments.

~~~powershell
.\llama-server.exe -m $weights --host 127.0.0.1 --port 8080 --no-webui `
  -c 262144 -np 1 -ngl 99 -ncmoe 16 -fit off -fa on -ctk q4_0 -ctv q4_0 `
  -b 3072 -ub 3072 -t 6 --load-mode none `
  --spec-type draft-dspark,ngram-simple --model-draft $draft --spec-draft-n-max 2 `
  --spec-draft-p-min 0.5 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 `
  --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 2
~~~

The locally patched llama.cpp banded build used these arguments with the same mainline-layout file.

~~~powershell
.\llama-server.exe -m $weights --host 127.0.0.1 --port 8080 --no-webui `
  -c 262144 -np 1 -ngl 99 --cpu-moe -fit off -fa on -ctk q4_0 -ctv q4_0 `
  -b 3072 -ub 3072 -t 6 --load-mode mmap --moe-cache-mib 3072 --moe-cache-cpu-share 0 `
  --spec-type draft-dspark,ngram-simple --model-draft $draft --spec-draft-n-max 2 `
  --spec-draft-p-min 0 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 `
  --spec-draft-device CUDA0 --spec-draft-ngl all --spec-draft-n-ubatch 64 `
  --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 2 --timeout 1800
~~~

</details>

## Laptop profile

Windows, RTX 4060 Laptop GPU 8 GB, Core i7-13700H, 32 GB RAM. The November 23, 2026 runs used 65,536- and 131,072-token Q4_0 GPU pools with a 2,560-token prefill chunk, tiled Q4_0 decode, and staged experts. The workload requested three exact record values at roughly 10%, 50% and 94% of each prompt. Both runs returned all three values.

[Launch settings and retrieval workload](inkling-ada-laptop.md)

## Exact model files

All runs use Inkling 2 Mini text weights quantized as **UD-IQ4_XS**.

| Engine | File and revision | SHA-256 |
| :--- | :--- | :--- |
| Inkling Ada and NInfer-all | Locally converted `Inkling-2-Mini-UD-IQ4_XS-visionq8.ninfer`, 16,494,910,976 bytes. The conversion uses the Unsloth root GGUF revision below. | `cdaa3d20474fb0874a4f899d29cd04e43993e32e16736d558d3fd3f71132d924` |
| Strata | `Inkling-2-Mini-UD-IQ4_XS.gguf` from Unsloth revision `ae9327c5fbc7333116ba32eee78bb5d2bbefae4f`. | `a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1` |
| llama.cpp | `Mainline_Support/Inkling-2-Mini-UD-IQ4_XS.gguf_file` from Unsloth revision `13bbdd720e18cf4e6be1d955f4db7b80446dde12`. The mainline layout preserves the quantized tensor values, renames loop tensors and groups the message-query rows by layer. | `374970bf6f151fc50da49419751c3e77b82e373a3b9445ded617bd386cd12163` |

The same external DSpark Q8_0 GGUF was passed to Strata and llama.cpp. Its SHA-256 is `2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773`. Inkling Ada uses the DSpark component in its packed artifact.

[Unsloth root GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/ae9327c5fbc7333116ba32eee78bb5d2bbefae4f/Inkling-2-Mini-UD-IQ4_XS.gguf) · [Mainline layout](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/13bbdd720e18cf4e6be1d955f4db7b80446dde12/Mainline_Support/README.md)

[Back to the benchmark tables](../README.md#benchmarks)
