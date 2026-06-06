# oxide-circuit-breaker

> A production-grade circuit breaker for GPU kernels, built in Rust. Protect your inference fleet from cascading failures with ternary-state semantics, automatic fallback routing, and CRDT-based fleet-wide synchronization.

---

## The Story: Why GPUs Need Circuit Breakers

Modern AI workloads run on fleets of accelerators—dozens, hundreds, or thousands of GPUs crunching through attention layers, reductions, and custom kernels in parallel. When one kernel starts failing—whether from a driver bug, a memory corruption, or a thermal throttle—the default behavior is to retry. And retry. And retry again. Each failed attempt wastes precious PCIe bandwidth, SM occupancy, and power. Worse, in a distributed setting, a sick node can poison the entire fleet as healthy nodes block waiting for stragglers.

The [circuit breaker pattern](https://martinfowler.com/bliki/CircuitBreaker.html), popularized in microservices, is just as essential at the kernel layer. `oxide-circuit-breaker` brings that resilience pattern directly into the GPU compute path. It tracks the health of individual kernels, trips open when failure thresholds are crossed, reroutes traffic to fallback implementations, and—critically—coordinates state across the entire fleet using conflict-free replicated data types (CRDTs) so that every node learns from every other node's experience.

This crate is part of the broader [SuperInstance](https://github.com/SuperInstance/SuperInstance) ecosystem, a collection of Rust libraries for building resilient, high-performance GPU infrastructure.

---

## Ternary-State Semantics

The breaker moves through three distinct states. Understanding them is the key to using the library effectively.

### Closed — The Happy Path

In the **Closed** state, the circuit is intact and traffic flows freely. Every call to the kernel is allowed through as `CallDecision::Execute`. Successes are counted, but more importantly, *consecutive* failures are tracked. If a kernel fails `threshold` times in a row, the breaker snaps **Open**. A single success resets the consecutive counter to zero, so intermittent hiccups do not cause unnecessary flapping.

### Open — Fail-Fast Protection

Once the breaker is **Open**, the kernel is considered unhealthy. Further calls are rejected immediately with `CallDecision::Rejected`, preventing wasted GPU cycles. If you configured a fallback kernel (for example, a slower but reliable CPU implementation), the breaker returns `CallDecision::Fallback` instead. The Open state is sticky: it persists until an external timer or orchestrator decides it is time to probe recovery.

### HalfOpen — Cautious Probing

When the orchestrator invokes `try_half_open()`, the breaker enters the **HalfOpen** state. A limited number of calls are admitted as `CallDecision::Probe`. If a probed call succeeds, the breaker closes immediately and normal operation resumes. If it fails even once, the breaker trips back to Open. This tight feedback loop ensures that recovery is validated with real traffic before the full fire-hose is re-opened.

---

## Quick Start

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
oxide-circuit-breaker = "0.1"
```

Then protect a kernel:

```rust
use oxide_circuit_breaker::{KernelBreaker, CallDecision};

fn main() {
    // Trip after 3 consecutive failures; fallback to a CPU kernel
    let mut breaker = KernelBreaker::with_fallback("flash_attention", 3, "attention_cpu");

    match breaker.allow_call() {
        CallDecision::Execute(kernel) => {
            // Run the fast GPU kernel
            let ok = run_kernel(&kernel);
            if ok { breaker.record_success(); } else { breaker.record_failure(); }
        }
        CallDecision::Fallback(kernel) => {
            // The fast path is down; use the reliable fallback
            run_kernel(&kernel);
            // Fallbacks do not automatically record success/failure;
            // you decide whether to heal the breaker based on fallback outcome.
        }
        CallDecision::Probe(kernel) => {
            // Half-open: cautiously try the real kernel again
            let ok = run_kernel(&kernel);
            if ok { breaker.record_success(); } else { breaker.record_failure(); }
        }
        CallDecision::Rejected => {
            // No fallback configured; drop or queue the work
            eprintln!("Kernel circuit is open; request rejected.");
        }
    }
}
```

---

## Fleet-Wide Awareness with CRDTs

A single-node circuit breaker is useful, but a GPU cluster is a distributed system. If `gpu-0` discovers that `custom_reduce_v2` is buggy, `gpu-1` through `gpu-7` should not have to rediscover that fact by crashing themselves.

`FleetBreakerState` solves this with a lightweight CRDT merge. Each node maintains a local map of kernel names to breaker states. When nodes gossip (over your transport of choice—RDMA, NCCL, or even a sidecar HTTP API), they exchange `FleetBreakerState` structs and call `merge()`:

```rust
use oxide_circuit_breaker::{FleetBreakerState, BreakerState};

let mut node_a = FleetBreakerState::new("gpu-0");
let mut node_b = FleetBreakerState::new("gpu-1");

// Node B observes a failure and marks the kernel Open
node_b.update("bad_kernel", BreakerState::Open);

// After gossip, Node A merges Node B's state
node_a.merge(&node_b);

assert_eq!(node_a.kernel_states["bad_kernel"], BreakerState::Open);
```

The merge logic is monotonic and conflict-free: **Open overrides HalfOpen, and either overrides Closed**. Versions are advanced with a max-plus-one strategy, so the data structure is idempotent and commutative. You can merge the same state twice, or merge out-of-order, and the result is always convergent.

---

## API Highlights

| Type / Method | Purpose |
|---------------|---------|
| `KernelBreaker::new(name, threshold)` | Create a breaker for a kernel with a failure threshold. |
| `KernelBreaker::with_fallback(name, threshold, fallback)` | Same, but route to a fallback kernel when Open. |
| `allow_call()` | Decide whether to execute, fallback, probe, or reject. |
| `record_success()` | Notify the breaker of a successful kernel execution. |
| `record_failure()` | Notify the breaker of a failure; may trip Open. |
| `try_half_open()` | Transition Open → HalfOpen to test recovery. |
| `failure_rate()` | Historical failure rate across all calls. |
| `FleetBreakerState::merge(other)` | CRDT merge for distributed state convergence. |

---

## Design Philosophy

1. **Zero async, zero allocations on the hot path.** `allow_call()`, `record_success()`, and `record_failure()` are simple enum matches and integer increments. They will not disturb your CUDA streams.
2. **Explicit over magical.** The library does not spawn threads, start timers, or call into drivers. You control when (and if) the breaker transitions from Open to HalfOpen.
3. **Fleet-first.** The CRDT layer means the breaker is useful in single-process tools *and* multi-node training jobs without changing the API.

---

## Testing

The crate ships with exhaustive unit tests covering threshold trips, fallback routing, consecutive-failure resets, half-open probes, and CRDT idempotency:

```bash
cargo test
```

All tests run without a GPU and complete in milliseconds.

---

## Relationship to SuperInstance

`oxide-circuit-breaker` is developed as a foundational resilience primitive for the [SuperInstance](https://github.com/SuperInstance/SuperInstance) project. SuperInstance provides orchestration, scheduling, and observability for large-scale GPU clusters; this crate handles the narrow but critical responsibility of keeping individual kernels from taking down a job. If you are building GPU infrastructure in Rust, we encourage you to explore the broader SuperInstance ecosystem.

---

## License

Licensed under the Apache License, Version 2.0. See LICENSE for details.

---

*Built with 🦀 for the next generation of GPU infrastructure.*
