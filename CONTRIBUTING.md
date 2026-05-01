# Contributing to ai-monitoring-dashboards

MIT licensed, contributions welcome. Particularly useful contributions: new dashboard JSON for additional model providers, alerting rules for common failure modes, and documentation for non-Docker setups.

## How to Contribute

1. Fork the repo on GitHub.
2. Create a topic branch off `main` (e.g. `feat/cohere-dashboard`).
3. Verify the stack still spins up cleanly with `docker compose up --build` and that any new dashboards render with the included demo data.
4. Open a pull request describing what you've added or changed.

## Development setup

The Python collector lives under `src/`. Install with the dev extras:

```bash
pip install -e ".[dev]"
```

The Grafana dashboards live under `dashboards/` (JSON) and are auto-provisioned. The Prometheus config lives under `prometheus/`.

## Code style

Python code is checked with [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check src/ tests/
```

JSON dashboards: please keep them formatted (most editors will do this on save). Don't include user-specific datasource UIDs — use the templated values.

## Testing

```bash
pytest -v
```

If your change touches the docker-compose stack, please confirm `docker compose up --build` still gets to "ready" within roughly a minute on a fresh checkout.

## Questions

Open an issue with the `question` label.
