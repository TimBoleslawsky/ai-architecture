# Predictive Maintenance — AI & Data Analytics

> Remaining-useful-life (RUL) prediction on industrial equipment, deployed as an
> online API **and** a scheduled batch scorer.

This project demonstrates the **AI & Data Analytics** workflow (taxonomy bucket 1):
raw sensor time-series → an **efficacy-first** deep model → model registry →
**two** serving patterns (on-demand + batch) → Postgres.

Everything runs on a laptop with `uv` + Docker Compose.

## Quickstart

```bash
uv sync                      # installs deps incl. dev (PyTorch/Lightning for training)
uv run pytest                # unit tests (serving tests skip until a model is trained)

uv run python -m src.data.prepare            # generate data -> windows + features
uv run python -m src.models.baseline         # (optional) RandomForest baseline comparison
uv run python -m src.models.engine.train \
  --config src/models/engine/config/rul_tcn.yaml   # train TCN, export ONNX, register

# online API (local, no Postgres needed)
uv run uvicorn src.serving.app:app --reload
curl http://127.0.0.1:8000/docs      # OpenAPI UI

# full stack: Postgres + API + batch worker
uv run python -m src.seed            # load sensor readings into Postgres
docker compose -f infra/docker-compose.yml up --build
```

A sample `/predict` call (last ~30 cycles of raw sensor readings):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"unit": 42, "rows": [{"cycle": 1, "s_1": 99.1, ...}, ...]}'
```

## Repository layout

```
src/data/       raw data (C-MAPSS downloader, synthetic generator, prepare)
src/features/   causal rolling-window features (used by the RF baseline)
src/models/
  engine/       faithful Lightning/OmegaConf engine (args, factories, runner,
                data module, TCN model, config yaml, train entry)
  baseline.py   RandomForest baseline comparison
src/serving/    FastAPI app, ONNX predictor, batch scorer
src/db.py       Postgres helpers (schema, seeding)
infra/          docker-compose.yml, schema.sql
docs/           arc42.md + ADRs
tests/          pytest suite
```