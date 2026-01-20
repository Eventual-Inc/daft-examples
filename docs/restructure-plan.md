# Repository Restructure Plan

## Final Structure

```
daft-examples/
├── quickstart/           # Entry point (5 files, run in 30 sec)
├── examples/             # Atomic examples, organized by feature
│   ├── prompt/           # LLM integration
│   ├── embed/            # Vectors & similarity
│   ├── classify/         # Classification
│   ├── io/               # File I/O
│   ├── files/            # daft.File abstraction
│   ├── udfs/             # Custom functions
│   └── sql/              # Window functions, joins
├── pipelines/            # End-to-end data pipelines
│   ├── rag/
│   ├── voice-analytics/
│   ├── code-search/
│   └── recommendations/
├── notebooks/            # Interactive exploration
└── tests/
```

## Migration Actions

1. Merge `patterns/` + `usage_patterns/` → `examples/`
2. Rename `use_cases/` → `pipelines/`
3. Delete empty `docs/`, `TEMPLATE/`, `models/`
4. Update README.md

## Rationale

Three levels: **learn** (quickstart) → **use** (examples) → **build** (pipelines)
