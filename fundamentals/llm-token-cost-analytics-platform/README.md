# LLM Token & Cost Analytics Platform

A real-time analytics platform that tracks token usage, latency, and cost across multiple LLM providers. The system collects request metrics through an API, stores them in a database, and visualizes usage patterns through a Grafana dashboard.

## 1. Problem

Production AI systems using Large Language Models (LLMs) need observability to understand:

- **Cost Management**: LLM API calls can be expensive, and costs scale with token usage. Without tracking, organizations risk unexpected bills.
- **Performance Monitoring**: Latency varies significantly between models and can impact user experience.
- **Usage Patterns**: Understanding which models are used most, by whom, and when helps optimize resource allocation.
- **Prompt Optimization**: Identifying expensive prompts allows teams to optimize token usage.

This platform solves the observability gap by providing real-time metrics for LLM operations.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM Analytics Platform                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐      ┌──────────────┐       ┌──────────────┐      │
│  │   Clients    │      │   FastAPI    │       │   SQLite     │      │
│  │  (Apps/SDKs) │─────▶│    Server    │─────▶│   Database   │      │
│  └──────────────┘      └──────────────┘       └──────────────┘      │
│                               │                     ▲               │
│                               ▼                     │               │
│                      ┌──────────────┐               │               │
│                      │   Grafana    │───────────────┘               │
│                      │  Dashboard   │                               │
│                      └──────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| API Server | FastAPI | REST API for tracking requests and analytics |
| Database | SQLite | Persistent storage with SQLAlchemy ORM |
| Dashboard | Grafana | Visualization and monitoring |
| Cost Engine | Python | Per-model pricing calculation |

### Data Flow

1. **Ingestion**: Client sends POST request to `/track` with metrics
2. **Processing**: Cost calculated using model-specific pricing
3. **Storage**: Record persisted to SQLite database
4. **Analysis**: Analytics endpoints aggregate data
5. **Visualization**: Grafana queries API for dashboard

## 3. Demo

### Starting the Platform

```bash
# Activate virtual environment
venv\Scripts\activate.bat

# Start API server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Tracking a Request

```bash
curl -X POST http://localhost:8000/track \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "input_tokens": 1200,
    "output_tokens": 400,
    "latency_ms": 2100,
    "user_id": "user123"
  }'
```

### Querying Analytics

```bash
# Get dashboard summary
curl http://localhost:8000/analytics/summary

# Cost by model
curl http://localhost:8000/analytics/cost/by-model

# Token statistics
curl http://localhost:8000/analytics/tokens
```

### Generating Sample Data

```bash
# Generate 1000 sample requests over 30 days
python scripts/generate_sample_data.py -n 1000 -d 30
```

### Grafana Dashboard

1. Install Grafana
2. Add SimpleJSON datasource pointing to `http://localhost:8000`
3. Import `dashboards/llm_analytics_dashboard.json`

## 4. Key Engineering Challenges

### Cost Calculation Accuracy

**Challenge**: Different models have different pricing structures (input vs output tokens).

**Solution**: Implemented per-model pricing in [`src/cost_calculator.py`](src/cost_calculator.py:35) using:
- Model-specific input/output rates per million tokens
- Fallback to default pricing for unknown models
- Fuzzy matching for model variants (e.g., "gpt-4" matches "gpt-4-turbo")

### Real-time Analytics Performance

**Challenge**: Computing percentiles over large datasets is expensive.

**Solution**: Used SQLite aggregation functions and in-memory sorting for percentile calculations in [`src/token_analyzer.py`](src/token_analyzer.py:75).

### Scalability

**Challenge**: Single database file may become a bottleneck.

**Solution**: 
- Database indexes on frequently queried columns (model, timestamp, user_id)
- Batched commits for bulk inserts
- SQLite is suitable for small-to-medium workloads; PostgreSQL migration path exists

## 5. Benchmarks

### Sample Data Generation

| Requests | Time | Database Size |
|----------|------|---------------|
| 100 | <1s | ~50KB |
| 1,000 | ~3s | ~500KB |
| 10,000 | ~30s | ~5MB |

### API Response Times

| Endpoint | Avg Response |
|----------|--------------|
| POST /track | ~5ms |
| GET /analytics/summary | ~10ms |
| GET /analytics/cost/by-model | ~15ms |
| GET /analytics/tokens | ~20ms |

## 6. Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| SQLite vs PostgreSQL | Simpler setup but less concurrency; suitable for single-server deployment |
| In-memory percentiles | Accurate but O(n log n) for large datasets |
| SimpleJSON datasource | Works but not real-time; consider Prometheus for production |
| Hardcoded pricing | Easy to extend but requires code changes for new models |

## 7. Future Work

### High Priority
- [ ] Add Prometheus metrics export for better Grafana integration
- [ ] Implement caching for frequent analytics queries
- [ ] Add rate limiting to prevent abuse

### Medium Priority
- [ ] Migrate to PostgreSQL for production use
- [ ] Add authentication and API keys
- [ ] Support dynamic pricing updates from external API

### Lower Priority
- [ ] Add alerting rules for cost thresholds
- [ ] Implement request/response logging for debugging
- [ ] Add support for streaming token counts
- [ ] Create a React frontend

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/track` | POST | Track new LLM request |
| `/requests` | GET | List tracked requests |
| `/analytics/summary` | GET | Dashboard summary |
| `/analytics/cost/by-model` | GET | Cost breakdown by model |
| `/analytics/cost/by-day` | GET | Daily cost trends |
| `/analytics/cost/by-user` | GET | Cost per user |
| `/analytics/tokens` | GET | Token statistics |
| `/analytics/latency` | GET | Latency statistics |
| `/analytics/models/performance` | GET | Model performance metrics |

---

Built with FastAPI, SQLAlchemy, and Grafana
