# Context DSL Coverage Matrix

## Key

- `R` = ordinary relational or DataFrame operators
- `S` = `segment`
- `A` = `annotate`
- `P` = `pack`
- `I` = `iterate`

Status labels:

- `clean` = natural fit for the current kernel
- `awkward` = expressible, but the abstraction leaks or needs helper sugar
- `break` = the kernel hides essential semantics or forces an imperative escape

The matrix is intentionally mixed:

- repo-backed witnesses where the repository already has an example
- external patterns where the repo is still thin

## Segmentation And Unit Formation

| # | Pattern | Witness | Desugaring | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 1 | Fixed-size chunking | `pipelines/context_engineering/chunking_strategies.py` | `S` | clean | Direct document-to-window split with stable provenance. |
| 2 | Sentence chunking | `pipelines/context_engineering/chunking_strategies.py` | `S` | clean | Same kernel as fixed windows, different boundary strategy. |
| 3 | Paragraph chunking | `pipelines/context_engineering/chunking_strategies.py`, `datasets/common_crawl/text_deduplication.py` | `S` | clean | Strong witness that chunking remains ordinary tabular expansion. |
| 4 | Structure-aware chunking | `pipelines/context_engineering/lambda_mapreduce.py` | `S + R` | clean | Pages already behave like document-structure units. |
| 5 | Semantic breakpoint chunking | external | `S + A(embed/boundary_score) + R(window/cut)` | awkward | The kernel handles it, but adjacency-aware cut logic is fussy enough to want helper sugar. |
| 6 | Sliding-window overlap bundles | external | `S + P(overlap)` | clean | Natural extension of chunking once `pack` can form bounded overlapping contexts. |

## Retrieval, Selection, And Packing

| # | Pattern | Witness | Desugaring | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 7 | Dense top-k retrieval | `pipelines/context_engineering/few_shot_example_selection.py`, `pipelines/context_engineering/arxiv_search/daily_workflow.py` | `A(embed) + R(similarity_join/sort/limit)` | clean | Retrieval is just scoring plus top-k once vectors are first-class columns. |
| 8 | Lexical or sparse retrieval | external | `A(score) + R(sort/limit)` | clean | No new context primitive is needed. |
| 9 | Hybrid retrieval fusion | external | `A(score_dense) + A(score_sparse) + R(union/group/sort)` | clean | Fusion is ordinary relational aggregation over scored candidates. |
| 10 | Query expansion retrieval | external | `S(subqueries) + retrieval + R(union/dedup)` | clean | Subqueries behave like another segmented axis. |
| 11 | Few-shot example selection | `pipelines/context_engineering/few_shot_example_selection.py` | `A(embed/distance) + R(rank/filter) + P` | clean | Strong proof that example selection is packing, not bespoke prompt code. |
| 12 | Parent-child retrieval | external | `S(parent_child) + retrieval + R(join) + P` | clean | Retrieve on child units, pack on parent units. |
| 13 | LLM relevance filtering | `pipelines/context_engineering/lambda_mapreduce.py` | `A(relevance) + R(filter)` | clean | Clear witness in `search` and `qa`. |
| 14 | Pairwise reranking | external | `A(score) + R(sort/limit)` | clean | Row-wise rescoring fits `annotate` directly. |
| 15 | Listwise reranking | external | `R(group) + P(candidate_list) + A(list_score)` | awkward | Setwise scoring leaks through the row-wise `annotate` abstraction. |
| 16 | Contextual compression under budget | `pipelines/context_engineering/lambda_mapreduce.py`, `pipelines/context_engineering/few_shot_example_selection.py` | `A(importance) + R(sort/filter) + P` | clean | This is one of the strongest arguments for `pack` as a kernel primitive. |
| 17 | MMR or diversity-aware selection | external | `A(relevance) + I(marginal_gain_step)` | awkward | Sequential marginal gain is not a simple sort. |
| 18 | Token-budget knapsack packing | external | `A(tokens/utility) + P` or `I(selector_step)` | awkward | Exact subset optimization should not be smuggled into a narrow `pack` without an explicit scope expansion. |

## Synthesis And Reasoning

| # | Pattern | Witness | Desugaring | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 19 | Map-reduce summarization | `pipelines/context_engineering/lambda_mapreduce.py` | `S + A(local_summary) + P + A(merge)` | clean | The canonical witness case in the repo. |
| 20 | Hierarchical summarization | `pipelines/context_engineering/lambda_mapreduce_pdf_iceberg.py` | `S + A + P + A + I(levels)` | clean | Multi-level reduction still feels like the same algebra, not a new one. |
| 21 | Query-focused QA synthesis | `pipelines/context_engineering/lambda_mapreduce.py` | retrieval + `A(partial_answer) + P + A(synthesize)` | clean | Another direct witness that synthesis is just packing plus annotation. |
| 22 | Citation-grounded answering | external | retrieval + `P(evidence_bundle) + A(answer_with_citations)` | clean | The important constraint is provenance, which the kernel already preserves. |
| 23 | Conflict resolution or source reconciliation | external | `R(group_by_claim) + A(stance) + P + A(resolve)` | awkward | Requires robust grouping keys and explicit contrast, which is still expressible but not elegant. |
| 24 | Structured extraction | `pipelines/key_moments_extraction.py` | `A(schema_struct) + R(explode/unnest)` | clean | Strong witness that typed outputs sit naturally inside the model. |

## Evaluation And Consensus

| # | Pattern | Witness | Desugaring | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 25 | Pairwise judging | `pipelines/context_engineering/llm_judge_elo.py` | `R(pair_construction) + P(judge_prompt) + A(verdict)` | clean | The judging step itself is not a breakpoint. |
| 26 | Self-consistency or majority vote | `pipelines/context_engineering/lambda_mapreduce.py` | repeated `A(sample_or_label) + R(group/count/sort)` | clean | Voting remains ordinary aggregation. |
| 27 | Conversation memory compaction | external | `S(turns) + A(salience) + P + optional I(compact_rounds)` | clean | The kernel covers both sliding-window and summarize-old patterns. |

## Iteration, Clustering, And Boundaries

| # | Pattern | Witness | Desugaring | Status | Note |
| --- | --- | --- | --- | --- | --- |
| 28 | Multi-hop retrieve-read-requery | external | `I(retrieve -> A(subquery) -> retrieval -> R(union/dedup) -> P)` | awkward | Frontier management is real but still table-shaped. |
| 29 | Near-duplicate dedup and clustering | `datasets/common_crawl/text_deduplication.py` | `S + A(normalize/minhash) + R(block/join) + I(connected_components)` | awkward | Very helpful witness for `iterate`, and strong evidence for future `block` sugar. |
| 30 | ELO or online preference ranking | `pipelines/context_engineering/llm_judge_elo.py` | pairwise judge + global sequential update | break | The classic online update depends on shared mutable state and ordered updates; batch approximations may fit better, but this form should stay outside the core DSL. |

## Coverage Result

| Status | Count |
| --- | --- |
| clean | 22 |
| awkward | 7 |
| break | 1 |

## What The Awkward Cases Have In Common

- They score or optimize over **sets**, not just rows.
- They need explicit **frontier** or **marginal gain** state.
- They need **blocking** or **graph-like convergence** rather than simple selection.
- They need exact or near-exact **subset optimization** rather than ordinary budgeted packing.
- They often suggest sugar families, not kernel promotion.

## Strongest Conclusions

1. `segment`, `annotate`, and `pack` cover the center of gravity of practical context engineering.
2. `iterate` is justified by multiple families, not just one contrived example.
3. `text_deduplication.py` is helpful because it demonstrates fixed-point table workflows and exposes likely future sugar families such as `block`, `cluster`, and `canonicalize`.
4. The cleanest real breakpoint today is **global sequential state**, not LLM interaction itself.
