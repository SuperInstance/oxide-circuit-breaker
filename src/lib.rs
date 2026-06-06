//! # oxide-circuit-breaker
//!
//! Circuit breaker pattern for GPU kernels.
//! Ternary state: Closed(healthy) / Open(failed) / HalfOpen(probing).
//! CRDT sync across GPU nodes for fleet-wide awareness.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState { Closed, Open, HalfOpen }

#[derive(Debug, Clone)]
pub struct KernelBreaker {
    pub kernel_name: String,
    pub state: BreakerState,
    pub success_count: u64,
    pub failure_count: u64,
    pub consecutive_failures: u32,
    pub threshold: u32,
    pub half_open_calls: u32,
    pub fallback: Option<String>,
    pub total_trips: u64,
}

impl KernelBreaker {
    pub fn new(name: &str, threshold: u32) -> Self {
        Self { kernel_name: name.into(), state: BreakerState::Closed,
            success_count: 0, failure_count: 0, consecutive_failures: 0,
            threshold, half_open_calls: 0, fallback: None, total_trips: 0 }
    }

    pub fn with_fallback(name: &str, threshold: u32, fallback: &str) -> Self {
        let mut b = Self::new(name, threshold);
        b.fallback = Some(fallback.into());
        b
    }

    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.consecutive_failures = 0;
        if self.state == BreakerState::HalfOpen {
            self.state = BreakerState::Closed;
            self.half_open_calls = 0;
        }
    }

    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.consecutive_failures += 1;
        if self.state == BreakerState::Closed && self.consecutive_failures >= self.threshold {
            self.state = BreakerState::Open;
            self.total_trips += 1;
        } else if self.state == BreakerState::HalfOpen {
            self.state = BreakerState::Open;
            self.total_trips += 1;
        }
    }

    /// Try to allow a call. Returns which kernel to actually call (may be fallback).
    pub fn allow_call(&mut self) -> CallDecision {
        match self.state {
            BreakerState::Closed => CallDecision::Execute(self.kernel_name.clone()),
            BreakerState::Open => {
                if let Some(ref fb) = self.fallback {
                    CallDecision::Fallback(fb.clone())
                } else {
                    CallDecision::Rejected
                }
            }
            BreakerState::HalfOpen => {
                self.half_open_calls += 1;
                CallDecision::Probe(self.kernel_name.clone())
            }
        }
    }

    /// Transition Open → HalfOpen after cooldown.
    pub fn try_half_open(&mut self) -> bool {
        if self.state == BreakerState::Open {
            self.state = BreakerState::HalfOpen;
            self.half_open_calls = 0;
            true
        } else { false }
    }

    pub fn failure_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 { 0.0 } else { self.failure_count as f64 / total as f64 }
    }

    pub fn is_healthy(&self) -> bool { self.state == BreakerState::Closed }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallDecision { Execute(String), Fallback(String), Probe(String), Rejected }

/// Fleet-wide breaker state synced via CRDT.
#[derive(Debug, Clone)]
pub struct FleetBreakerState {
    pub node_id: String,
    pub kernel_states: HashMap<String, BreakerState>,
    pub version: u64,
}

impl FleetBreakerState {
    pub fn new(node_id: &str) -> Self {
        Self { node_id: node_id.into(), kernel_states: HashMap::new(), version: 0 }
    }

    pub fn update(&mut self, kernel: &str, state: BreakerState) {
        self.kernel_states.insert(kernel.into(), state);
        self.version += 1;
    }

    /// CRDT merge: Open overrides Closed (fail-fast propagation).
    pub fn merge(&mut self, other: &FleetBreakerState) {
        for (kernel, state) in &other.kernel_states {
            let current = self.kernel_states.get(kernel).copied();
            let merged = match (current, state) {
                (Some(BreakerState::Open), _) | (_, BreakerState::Open) => BreakerState::Open,
                (Some(BreakerState::HalfOpen), _) | (_, BreakerState::HalfOpen) => BreakerState::HalfOpen,
                _ => BreakerState::Closed,
            };
            self.kernel_states.insert(kernel.clone(), merged);
        }
        self.version = self.version.max(other.version) + 1;
    }

    pub fn open_kernels(&self) -> Vec<&str> {
        self.kernel_states.iter()
            .filter(|(_, s)| **s == BreakerState::Open)
            .map(|(k, _)| k.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closed_allows() {
        let mut b = KernelBreaker::new("attention", 3);
        assert!(matches!(b.allow_call(), CallDecision::Execute(_)));
    }

    #[test]
    fn test_trips_on_threshold() {
        let mut b = KernelBreaker::new("reduce", 3);
        for _ in 0..3 { b.record_failure(); }
        assert_eq!(b.state, BreakerState::Open);
        assert!(b.total_trips > 0);
    }

    #[test]
    fn test_fallback_on_open() {
        let mut b = KernelBreaker::with_fallback("reduce", 2, "reduce_cpu");
        b.record_failure(); b.record_failure();
        let decision = b.allow_call();
        assert!(matches!(decision, CallDecision::Fallback(_)));
    }

    #[test]
    fn test_success_resets_consecutive() {
        let mut b = KernelBreaker::new("filter", 3);
        b.record_failure(); b.record_failure();
        b.record_success();
        assert_eq!(b.consecutive_failures, 0);
    }

    #[test]
    fn test_half_open_probe() {
        let mut b = KernelBreaker::new("kernel", 2);
        b.record_failure(); b.record_failure(); // trip
        b.try_half_open();
        assert_eq!(b.state, BreakerState::HalfOpen);
        assert!(matches!(b.allow_call(), CallDecision::Probe(_)));
    }

    #[test]
    fn test_half_open_success_closes() {
        let mut b = KernelBreaker::new("kernel", 2);
        b.record_failure(); b.record_failure();
        b.try_half_open();
        b.record_success();
        assert_eq!(b.state, BreakerState::Closed);
    }

    #[test]
    fn test_crdt_merge_propagates_open() {
        let mut node1 = FleetBreakerState::new("gpu-0");
        let mut node2 = FleetBreakerState::new("gpu-1");
        node2.update("kernel", BreakerState::Open);
        node1.merge(&node2);
        assert_eq!(node1.kernel_states["kernel"], BreakerState::Open);
    }

    #[test]
    fn test_crdt_merge_idempotent() {
        let mut n1 = FleetBreakerState::new("a");
        n1.update("k", BreakerState::Closed);
        let n2 = n1.clone();
        n1.merge(&n2);
        assert_eq!(n1.kernel_states["k"], BreakerState::Closed);
    }
}
