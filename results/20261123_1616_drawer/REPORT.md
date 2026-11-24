# Run 20261123_1616_drawer

task `drawer_lying`, condition `text`, phase `report`, created 2026-11-23T16:16:37

## Cost and time

cost $73.20 (agents $68.90, judge $4.30 / 60 calls)  tokens 1k in / 787k out (+149.5M cache read, +2860k cache write)  agent 184 min (619 turns)  robot 390 min  gpu 0 min  wall 10.4 h

Per call: `usage.jsonl` (agent turns, VLM judge calls, GPU jobs, evaluations, each with the running totals).

## Agents

| role | status | runs | robot runs | first success (robot attempt) | time to first success | turns | tokens in / out / cached | cost | agent time | repairs |
|---|---|---|---|---|---|---|---|---|---|---|
| reset | done | 37 | 18 | 2 | 14.6 min | 238 | 358 / 311726 / 53429232 | $27.99 | 97 min | 2 |
| execute | done | 27 | 9 | 8 | 131.2 min | 381 | 564 / 475667 / 96053113 | $40.90 | 87 min | 0 |

## Collection

- episodes: 0 (success 0, fail 0, unsure 0; 0%)
- unsure episodes deleted and redone: 0
- resets: 77 (ok 67, fail 10, manual 0)
- rounds: 0
- VLM judge: 60 calls, $4.30
- operator review queue: 0 pending

## Datasets and training

- no datasets
- no training runs

## Evaluations

| eval | checkpoint | episodes | success | rate | met | DAgger corrections (success/total) |
|---|---|---|---|---|---|---|

deployable checkpoint: `-`
