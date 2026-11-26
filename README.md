# A Taxonomy of AI Use Cases for Industrial Companies

> A taxonomy of AI use cases for industrial companies, defined by the
> **requirements the use case places on the company** — not by the task itself.
> Each bucket cleanly separates the *company demands*; when tasks overlap, so do
> the requirements.

## Why Requirements, not Tasks

A company rarely needs to "classify tasks" — it needs to *build capabilities*.
Defining the taxonomy by company requirements makes it stable and actionable:

- Each bucket names what you must build, staff and operate.
- Each bucket has a distinct **cost driver** (where the money and the hard
  problems actually go).
- The requirement sets are largely **additive**, so hybrid use cases are easy to
  reason about: a task that overlaps two buckets demands the union of both
  capability sets.

---

## The three Buckets

### Bucket 1 — Analytics: *predict, then decide*

Predictive models turn structured data into insight that drives decisions
(forecasting, churn, pricing, predictive maintenance, quality analytics).

**The company must have**
- **Data foundation** — ingestion, data quality, lineage, warehousing/lakehouse, feature store
- **Analytical engineering** — reproducible training, evaluation, model registry
- **Decision layer** — dashboards/alerts, KPIs, and *people empowered to act on predictions*
- **Model ops** — drift monitoring, retraining cadence

**Roles:** data engineers, data scientists, analysts, BI.

**Cost drivers & typical failure modes:** bad data infrastructure — no clean data flow means no trustworthy predictions, whatever the model.

**Example in this repo:** [`01-predictive-maintenance/`](01-predictive-maintenance/) — sensor-fleet RUL prediction: efficacy-first TCN (deep, cloud) vs RandomForest baseline → ONNX → online API + batch scorer.

### Bucket 2 — Automation: *do the tedious thing, correctly, at scale*

Automation systems (LLMs and rules) take well-defined, repetitive, high-volume
work off people's plates — and take an action in a workflow (invoice
processing, support triage, document routing).

**The company must have** 
1. **Well-defined workflows & processes** — automation only works on defined
   processes; standardize first (scattered legacy documentation is a blocker,
   not a nuisance)
2. **Integration capabilities & infrastructure** — connectors, APIs, queues,
   idempotency, workflow observability — plus the runtime to run models,
   including **hosting your own LLM** when data privacy requires it
3. **Governance & observability of models/agents** — schema enforcement,
   deterministic validation, human-in-the-loop, audit trails, tracing/monitoring
4. **Change management** — vendor governance, data protection, liability, training

**Roles:** AI engineers, integration/backend engineers, process owners, compliance.

**Cost drivers & typical failure modes:** integration/ guardrails/ change management (the model is the cheap part) + there is no clearly defined process to automate (e.g. legacy documentation scattered across a hundred places) — you must standardize before you automate.

**Example in this repo:** [`02-llm-automation/`](02-llm-automation/) — invoice processing with a small open-source LLM, guardrails and human-in-the-loop.

\* Agentic AI (multi-step, tool-using agents) is an *autonomy extension* of this bucket — same cost driver, amplified, plus tool/action safety, trajectory evaluation and cost/runaway control.

### Bucket 3 — AI as a product: *the model is the feature*

The model lives inside a product and enables new functionality for end-users in
the real world (self-driving, quality inspection on the line, embedded
assistants). It usually runs live, often on the edge.

**The company must have**
- **Model optimization engineering** — quantization, pruning, ONNX/TFLite, latency/power budgets
- **Real-time / edge engineering** — deployment to devices, OTA updates, versioning, rollback
- **Robustness & safety** — edge-case/adversarial testing, graceful degradation, field monitoring, safety cases
- **Domain data ops** — collecting and annotating data *where the product lives*
- **Cross-functional product org** — hardware/software co-design, product management

**Roles:** ML engineers, electrical engineers, product managers, QA/safety.

**Cost drivers & typical failure modes:** computational restraints on edge — the model doesn't fit the device's latency, power or memory budget. And if it does, it's performance does not justify the engineering costs.

**Example in this repo:** [`03-quality-inspection/`](03-quality-inspection/) — steel-defect classification shipped as a product: ONNX API + web UI + prediction telemetry.

---

## Individual Project Details

### Quickstart (each project)

```bash
# Project 1 - predictive maintenance
cd 01-predictive-maintenance
uv sync
uv run python -m src.data.prepare
uv run python -m src.models.baseline         # (optional) baseline comparison
uv run python -m src.models.engine.train \
  --config src/models/engine/config/rul_tcn.yaml
uv run uvicorn src.serving.app:app --reload        # POST /predict

# Project 2 - invoice automation (needs Ollama)
cd 02-llm-automation
uv sync
uv run uvicorn src.serving.app:app --reload        # POST /invoices/process
# or: docker compose -f infra/docker-compose.yml up --build

# Project 3 - defect inspection
cd 03-quality-inspection
uv sync
uv run python -m src.data.synthetic --per-class 150
uv run python -m src.models.train --epochs 12
uv run uvicorn src.serving.app:app --reload        # web UI on :8000
```

Every project also ships a `docker compose` deployment and its own `README`.

### What's inside each project

Each project follows the same layout so the arc42/C4/ADR story is comparable:

```
<project>/
├── docs/
│   ├── arc42.md          # full arc42 doc with C4 diagrams (mermaid)
│   └── adr/              # numbered ADRs capturing the trade-offs
├── src/                  # data, features, models, serving
├── infra/                # docker-compose.yml, schema.sql
├── tests/                # pytest suite (uv run pytest)
├── pyproject.toml        # uv-managed dependencies (+ uv.lock)
└── README.md             # how to run end-to-end
```