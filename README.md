# oxide-circuit-breaker

Circuit breaker pattern for GPU kernels. Track success/failure rates, trip on threshold, reroute to fallback. Ternary state: Closed/Open/HalfOpen with CRDT sync across GPU nodes.

## Why This Matters

# oxide-circuit-breaker
Circuit breaker pattern for GPU kernels.
Ternary state: Closed(healthy) / Open(failed) / HalfOpen(probing).
CRDT sync across GPU nodes for fleet-wide awareness.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum BreakerState
pub struct KernelBreaker
pub fn new
pub fn with_fallback
pub fn record_success
pub fn record_failure
pub fn allow_call
pub fn try_half_open
pub fn failure_rate
pub fn is_healthy
pub enum CallDecision
pub struct FleetBreakerState
```

## Usage

```toml
[dependencies]
oxide-circuit-breaker = "0.1.0"
```

```rust
use oxide_circuit_breaker::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-circuit-breaker.git
cd oxide-circuit-breaker
cargo test    # 8 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Rust | 206 |
| Public API | 16 items |

## License

Apache-2.0
