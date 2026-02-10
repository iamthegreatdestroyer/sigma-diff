# Distributed Tracing Guide - Ryzanstein LLM

## Overview

This guide covers the distributed tracing implementation for Ryzanstein LLM, enabling end-to-end request visibility across all inference components.

## Quick Start

### 1. Start Observability Stack

```bash
cd PHASE2_DEVELOPMENT/docker
docker-compose -f docker-compose.observability.yaml up -d
```

### 2. Access Dashboards

- **Jaeger UI**: http://localhost:16686
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/ryzanstein)
- **Kibana**: http://localhost:5601

### 3. Instrument Code

```python
from tracing.tracer import Tracer
from tracing.context import TraceContext

# Initialize tracer
tracer = Tracer(service_name="ryzanstein-inference")

# Create spans
with tracer.start_span("inference-request") as span:
    span.set_attribute("model", "bitnet-7b")
    span.set_attribute("tokens", 100)
    
    # Your inference code here
    result = model.generate(prompt)
    
    span.set_attribute("output_tokens", len(result))
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ API Server   │  │ Inference    │  │ GPU Coord    │          │
│  │              │  │ Engine       │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┼──────────────────┘                  │
│                            │                                     │
│                    ┌───────▼───────┐                            │
│                    │ OpenTelemetry │                            │
│                    │   SDK         │                            │
│                    └───────┬───────┘                            │
└────────────────────────────┼────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
       ┌──────▼─────┐ ┌──────▼─────┐ ┌─────▼──────┐
       │   Jaeger   │ │ Prometheus │ │    ELK     │
       │  (Traces)  │ │ (Metrics)  │ │   (Logs)   │
       └────────────┘ └────────────┘ └────────────┘
```

## Trace Context Propagation

### HTTP Headers

```
traceparent: 00-<trace-id>-<span-id>-<flags>
tracestate: ryzanstein=...
```

### gRPC Metadata

```python
metadata = [
    ("traceparent", context.to_traceparent()),
    ("x-request-id", request_id),
]
```

## Span Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| yzanstein.model | string | Model name |
| yzanstein.tokens.input | int | Input token count |
| yzanstein.tokens.output | int | Output token count |
| yzanstein.latency.prefill_ms | float | Prefill latency |
| yzanstein.latency.decode_ms | float | Decode latency |
| yzanstein.gpu.id | int | GPU used |
| yzanstein.batch.size | int | Batch size |

## Sampling Configuration

Edit configs/sampling-strategies.json:

```json
{
  "service_strategies": [
    {
      "service": "ryzanstein-inference",
      "type": "probabilistic",
      "param": 0.1
    }
  ]
}
```

## Troubleshooting

### Traces Not Appearing

1. Check Jaeger is running: docker ps | grep jaeger
2. Verify endpoint: curl http://localhost:14268/api/traces
3. Check sampling rate is not 0

### High Latency

1. Use batch span processor
2. Increase batch size
3. Reduce attribute count

## Performance Impact

- Span creation: ~0.1ms overhead
- Context propagation: ~0.05ms overhead
- Total overhead: <1% of request time

---

**Sprint 3.2 Complete** ✅
