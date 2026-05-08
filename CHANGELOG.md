# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-08

### Added

- Three pre-built Grafana dashboards: LLM API Monitor (token usage, cost tracking, latency percentiles, error rates), Model Performance (prediction accuracy, feature drift, inference latency, throughput), and Agent Orchestration (routing distribution, agent response times, supervisor iterations, fallback rates)
- Python Prometheus metrics library (`ai_monitor`) with three modules: `LLMMetrics` (requests, tokens, latency, USD cost), `ModelMetrics` (predictions, accuracy, drift), and `AgentMetrics` (routing, response duration, iterations)
- Context-manager and decorator helpers (`track_call`, `track_llm_call`) for automatic latency and token tracking around LLM calls
- One-command Docker Compose stack bundling Prometheus, Grafana, and a demo FastAPI app exposing `/metrics`
- Auto-provisioned Grafana datasource and dashboards (no manual import needed) accessible without login on startup
- Demo simulation endpoint (`/api/simulate`) for generating realistic LLM, model, and agent traffic patterns to populate the stack
- Configurable metric prefix (`METRICS_PREFIX`) and simulation cadence (`SIMULATION_INTERVAL`) via environment variables
- Documentation and code examples covering library integration, custom metric prefixes, and embedding the metrics modules in user applications
