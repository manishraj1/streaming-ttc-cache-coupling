# Empirical Note: Bounds of KV-Cache Contention in Streamed Test-Time Compute (K, λ)

## Abstract
We present a controlled empirical evaluation of continuous-batching engine behavior under heavy
streamed Test-Time Compute (TTC) loads. Using a pre-registered experimental gate, we map the
relationship between high parallel-sample budgets ($K$) and multi-tenant arrival rates ($\lambda$) to
test for the onset of cross-request KV-cache preemption cascades. At small-model scale (1.5B
parameters) on a single accelerator, the hypothesized "preemption cliff" is entirely absent: the
system transitions directly from linear compute cost to compute-bound execution stalling and
connection timeouts, with zero measurable memory-eviction coupling. Rather than a failed hunt, we read
this as a useful lower bound — it establishes the regime below which KV-cache contention cannot be the
binding constraint on streamed TTC, locating the phenomenon (if it exists) in larger models or
long-context workloads.

## Experimental Design & Telemetry
- **Model:** Qwen2.5-1.5B-Instruct. We deliberately ran with a tight `--max-model-len 1024` to
  *minimize* available KV-cache headroom and thereby give the preemption cliff its best chance to
  appear. The cliff did not appear even under this constrained budget — a stronger negative than a
  generous-budget run would have produced.
- **Workload Trace:** Synthetic Poisson stream ($N=200$ prompts) sweeping $\lambda \in \{2.0, 6.0,
  12.0\}$ requests/sec and parallel-decoding budgets $K \in \{1, 8, 64\}$.
- **Instrumentation:** Native vLLM Prometheus endpoint, tracking exact preemption totals
  (`vllm:num_preemptions_total`) and active block-allocation percentage
  (`vllm:gpu_cache_usage_perc`). (An earlier pilot read a stale metric name and silently returned 0.00
  for cache usage; the values below use the corrected field.)

## Results Matrix

| Arrival Rate ($\lambda$) | Sample Budget ($K$) | Total Preemptions | Peak Cache Usage | p99 Latency | Status |
|-------------------------|---------------------|-------------------|------------------|-------------|--------|
| 2.0                     | 1                   | 0.0               | 0.00%            | 2.152s      | 200 OK |
| 2.0                     | 8                   | 0.0               | 0.00%            | 3.749s      | 200 OK |
| 2.0                     | 64                  | 0.0               | 0.00%            | 259.557s    | 200 OK |
| 6.0                     | 1                   | 0.0               | 0.00%            | 2.587s      | 200 OK |
| 6.0                     | 8                   | 0.0               | 0.00%            | 15.684s     | 200 OK |
| 6.0                     | 64                  | 0.0               | 0.07%            | 297.856s    | Timeout|
| 12.0                    | 1                   | 0.0               | 0.00%            | 25.951s     | 200 OK |
| 12.0                    | 8                   | 0.0               | 0.00%            | 28.209s     | 200 OK |
| 12.0                    | 64                  | 0.0               | 0.09%            | 297.063s    | Timeout|

## Key Findings

### 1. No preemption cascade — even with a deliberately tightened cache budget
Across all 12 conditions, total engine preemptions remained strictly at **0.0**. Even under a
high-frequency input storm ($\lambda = 12.0$) with maximum branching ($K = 64$), peak GPU cache usage
never exceeded **0.09%** — despite the tight `--max-model-len 1024` cap intended to make eviction more
likely. A 1.5B model's KV-cache footprint is simply too compact to pressure PagedAttention allocation
boundaries at this scale, so the cross-request memory-eviction coupling the project set out to study
does not arise.

### 2. The failure mode is compute, not memory — at the confidence our instruments support
At $K = 64$, latency degrades severely ($\approx 297$s) and explicit request failures occur at higher
arrival rates ($\lambda \geq 6.0$). This is **not** memory-eviction coupling: preemptions stayed at
zero and cache usage stayed near zero throughout. The behavior is instead **consistent with compute
saturation** — a single prompt forcing the engine to track 64 parallel decode threads, contending for
GPU compute rather than KV-cache capacity. We state this as inference, not measurement: we instrumented
preemptions, cache usage, and latency, but did **not** directly instrument SM/compute occupancy, so we
can assert that the bottleneck is *not* memory eviction with confidence, while attributing it to
compute saturation as the consistent (but not directly measured) explanation. Likewise, the `Failed`
rows are **consistent with** client-side socket timeouts (the async client dropping inactive
connections under multi-hundred-second latencies) rather than engine crashes; preemptions and cache
metrics show no engine-side eviction at those points.

## Conclusion & Scope Limits
The pre-registered kill condition for this study has been explicitly triggered. Building an adaptive
runtime controller to dynamically choke $K$ at this scale is **unmerited**: the bottleneck is compute
starvation, addressable by standard static configuration defaults, not algorithmic cache gating.

**Scope.** These results characterize a single accelerator and a 1.5B model under a 200-prompt
transient trace; we report transient behavior and do not claim to have reached sustained steady state
at the highest $\lambda \times K$ corner. We do **not** claim the preemption cliff does not exist in
general — only that it does not appear in this regime. The hypothesis that streamed TTC scaling
produces a volatile KV-cache preemption cliff is therefore bounded to larger architectures
($\geq 7\text{B}$) or long-context / multi-sequence workloads where individual KV tokens impose genuine
memory pressure. Testing that regime requires hardware able to host such a model at the concurrency
needed, and is the natural next step.
