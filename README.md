# oxide-circuit-breaker

> Ternary-state circuit breakers for GPU kernels, with CRDT-synchronized fleet awareness.

## Background Theory

GPU kernels fail in ways that differ from traditional microservices. A kernel may segfault on one input shape but not another. A memory access pattern may be correct on an A100 but trigger an illegal instruction on a V100. A driver update may introduce latency spikes that look like failures under tight deadlines. Naive retry logic under these conditions can generate thundering herds that saturate already-struggling nodes.

The circuit breaker pattern, introduced in distributed systems to prevent cascading failure, maps elegantly onto GPU kernel health:

- **Closed**: The kernel is healthy; calls execute normally.
- **Open**: The kernel has failed repeatedly; calls are rejected or routed to a fallback.
- **HalfOpen**: A probing window where a small number of calls are allowed to test recovery.

`oxide-circuit-breaker` extends this classic pattern in two ways. First, it encodes the three states as a **ternary logic** that mirrors the SuperInstance `{-1, 0, +1}` motif: Open, HalfOpen, Closed correspond naturally to negative, neutral/uncertain, and positive. Second, it propagates breaker state across the fleet via a **CRDT merge**, so that a kernel failure observed on one GPU node can protect the entire fleet.

## How It Works

### KernelBreaker

A `KernelBreaker` tracks the health of a single named kernel:

- `success_count` and `failure_count` record lifetime statistics.
- `consecutive_failures` drives the trip condition.
- `threshold` is the number of consecutive failures required to transition Closed → Open.
- `half_open_calls` counts probes in the HalfOpen state.
- `fallback` optionally names an alternative kernel to call when Open.
- `total_trips` records how many times the breaker has opened.

State transitions:

- **Closed → Open**: Consecutive failures reach threshold, or a HalfOpen probe fails.
- **Open → HalfOpen**: An external healing process calls `try_half_open()` after a cooldown.
- **HalfOpen → Closed**: A probe succeeds.
- **HalfOpen → Open**: A probe fails.

### CallDecision

When a caller asks `allow_call()`, the breaker returns one of four decisions:

- `Execute(name)` — Run the primary kernel.
- `Fallback(name)` — Run the fallback kernel (if configured).
- `Probe(name)` — Run a limited probe while HalfOpen.
- `Rejected` — Fail fast; no fallback available.

### FleetBreakerState

`FleetBreakerState` aggregates breaker states for all kernels on a node. Nodes can merge states using a CRDT rule where `Open` overrides `HalfOpen`, and `HalfOpen` overrides `Closed`. This produces **monotonic failure propagation**: once the fleet learns a kernel is failing, that knowledge cannot be accidentally overwritten by a node that has not yet observed the failure.

## Experiments

The test suite encodes the following claims:

```rust
#[test]
fn test_trips_on_threshold() {
    // Three consecutive failures trip Closed → Open.
}

#[test]
fn test_fallback_on_open() {
    // Open state routes to fallback kernel when configured.
}

#[test]
fn test_half_open_success_closes() {
    // A successful probe in HalfOpen returns the breaker to Closed.
}

#[test]
fn test_crdt_merge_propagates_open() {
    // Fleet-wide merge propagates Open state across nodes.
}
```

A larger experiment: simulate a 32-node fleet where one kernel has a 1% failure rate under a specific input distribution. Measure:

- Mean time to trip across the fleet.
- False-trip rate when failure rate is actually zero.
- Recovery latency under `try_half_open()` with exponentially increasing cooldown.
- Bandwidth saved by CRDT propagation vs. broadcasting every health update.

## Applications

- **Kernel-level fault isolation**: Prevent a buggy attention kernel from repeatedly crashing the host process.
- **Graceful degradation**: Route failed GPU kernels to CPU fallback implementations.
- **Fleet-wide outage containment**: One node observes a driver-level failure; all nodes stop calling the affected kernel within milliseconds.
- **Integration with `oxide-canary`**: Canary failures should trip breakers automatically to protect the baseline.
- **Integration with `oxide-fleet`**: The fleet coordinator can avoid assigning work to kernels marked Open on a given agent.

## Open Questions

1. **Threshold calibration**: Should thresholds be static per kernel, learned from historical failure rates, or adaptive to fleet-wide stress?
2. **Half-open timing**: Is exponential backoff the right cooldown model, or should recovery be driven by explicit health checks?
3. **Fallback fidelity**: When a GPU kernel falls back to CPU, how do we preserve numerical equivalence for scientific computing workloads?
4. **CRDT partition tolerance**: During a network partition, can nodes on either side make conflicting breaker decisions that harm convergence?

## Cross-Links

- [SuperInstance agent-knowledge / FAULT-TOLERANCE.md](https://github.com/SuperInstance/agent-knowledge/blob/main/FAULT-TOLERANCE.md) — Theoretical foundations for failure containment.
- [SuperInstance agent-knowledge / TERNARY-NUMBERS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TERNARY-NUMBERS.md) — Ternary framing of Closed/HalfOpen/Open.
- [SuperInstance agent-knowledge / AGENT-TO-AGENT-PROTOCOL.md](https://github.com/SuperInstance/agent-knowledge/blob/main/AGENT-TO-AGENT-PROTOCOL.md) — Protocol layer beneath CRDT sync.
- `oxide-fleet` — Uses breaker state in work assignment decisions.
- `oxide-canary` — Generates kernel version changes that may trip breakers.
- `oxide-constructs` — Supplies the kernels being protected.

## Quick Start

```rust
use oxide_circuit_breaker::{KernelBreaker, BreakerState, CallDecision, FleetBreakerState};

let mut breaker = KernelBreaker::with_fallback("attention", 3, "attention_cpu");

// Simulate failures.
breaker.record_failure();
breaker.record_failure();
breaker.record_failure();
assert_eq!(breaker.state, BreakerState::Open);

match breaker.allow_call() {
    CallDecision::Fallback(name) => println!("Routing to fallback: {}", name),
    _ => panic!("Expected fallback"),
}

// Simulate fleet-wide propagation.
let mut node_a = FleetBreakerState::new("gpu-0");
let mut node_b = FleetBreakerState::new("gpu-1");
node_b.update("attention", BreakerState::Open);
node_a.merge(&node_b);
assert_eq!(node_a.kernel_states["attention"], BreakerState::Open);
```
