# Eval report

## Ablation (A: schema only -> D: + repair)

| Config | Accuracy | Safety pass rate | Avg latency (s) | Cache hit ratio |
|---|---|---|---|---|
| A | 93% | 100% | 6.17 | 0% |
| B | 100% | 100% | 6.97 | 63% |
| C | 100% | 100% | 5.82 | 79% |
| D | 100% | 100% | 5.66 | 78% |

## Accuracy by category

| Category | A | B | C | D |
|---|---|---|---|---|
| adversarial | 100% | 100% | 100% | 100% |
| out_of_scope | 100% | 100% | 100% | 100% |
| segmented | 100% | 100% | 100% | 100% |
| simple | 100% | 100% | 100% | 100% |
| trap | 67% | 100% | 100% | 100% |

## Failures, config D

None.
