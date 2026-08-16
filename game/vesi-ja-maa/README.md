# Vesi & Maa

A browser game about selling land and water to datacenters in Finland, built as
an [Archetype](https://github.com/VangelisTech/archetype) world.

Open `index.html`. No build, no dependencies, no network calls.

## Why it looks like this

Archetype is a dataframe-first ECS on [Daft](https://www.daft.ai/): state lives in
components on entities, processors transform whole populations in priority order,
every tick persists as immutable history, and a fork continues from an earlier
state without overwriting the original run.

The game is structured the same way rather than merely themed that way:

| Archetype | Here |
|---|---|
| Entity | a parcel |
| Component | `water` (licence, draw), `grid` (MW, queue), `link` (fibre, satellite), `heat` (exchanger), `tenancy` (tenant, rent, term) |
| Processor | `Recharge` (10) → `GridQueue` (20) → `PermitOffice` (30) → `Settlement` (40) → `HeatRecovery` (50) → `Demand` (60) → `CouncilReview` (70) → `Analyst` (75) → `Ledger` (80) |
| Tick history | every quarter snapshotted; the ledger shows which processor wrote each line |
| Fork | scrub the history rail, fork a counterfactual; abandoned branches stay on the rail |

## The analyst is world state

The analyst is a component written by a processor, not a UI overlay. It reads the
same state every other processor reads, projects the aquifer forward using the
world's own hydrology constants, names the binding constraint, and writes its
verdict back into the tick. So it is scrubbable after the fact — you read what it
said at the time, not what it would say now — and it forks with everything else.

It is deliberately deterministic. That means it replays identically down a fork,
so when two branches disagree, the disagreement is attributable to your decisions
and nothing else. The panel shows the same tick read in every branch side by side.

## Model

Exogenous events and tenant arrivals are on a fixed schedule, and there is no
RNG anywhere, so a fork replays the same weather and the same buyers.

Hydrology, per quarter, with `σ` the seasonal coefficients:

```
Δaquifer = RECHARGE_BASE·σ_recharge(q) − Σ_i draw_i · DRAW_FACTOR · σ_cooling(q)
draw_i   = thirst_i · (0.22 if closed loop else 1)
```

`Recharge` and the analyst's projection read `RECHARGE_BASE` and `DRAW_FACTOR`
from the same constants, so the forecast cannot drift from the world.

Final position — money is worth only what the ground and the council let you keep:

```
score = (cash/3000) · (aquifer/70) · (standing/60) + 8·heatGWh
```

Municipalities are real. Parcels, tenants and hydrology are invented.

## Status

Balance-tested headlessly across several strategies. The scoring separates
careless play from careful play as intended. The loss conditions (aquifer
collapse, consent withdrawn) are reachable in principle but no scripted strategy
has yet driven the aquifer below ~33, so the thirsty line is not yet as punishing
as intended. Recurring site upkeep is not implemented, which is why insolvency is
currently unreachable.
