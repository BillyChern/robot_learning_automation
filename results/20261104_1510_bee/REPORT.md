# Run 20261104_1510_bee

task `bee_in_bowl`, condition `text`, phase `report`, created 2026-11-04T15:10:35

## Cost and time

cost $1644.24 (agents $1621.34, judge $22.90 / 678 calls)  tokens 5k in / 1263k out (+475.1M cache read, +295823k cache write)  agent 291 min (2662 turns)  robot 4557 min  gpu 825 min  wall 155.8 h

Per call: `usage.jsonl` (agent turns, VLM judge calls, GPU jobs, evaluations, each with the running totals).

## Agents

| role | status | runs | robot runs | first success (robot attempt) | time to first success | turns | tokens in / out / cached | cost | agent time | repairs |
|---|---|---|---|---|---|---|---|---|---|---|
| execute | done | 4 | 1 | 1 | 5.7 min | 821 | 1588 / 354378 / 96878502 | $709.09 | 103 min | 0 |
| reset | done | 75 | 69 | 2 | 14.0 min | 1467 | 2790 / 699878 / 325395062 | $814.21 | 129 min | 62 |
| learn | done | 0 | 0 | - | - | 374 | 682 / 208302 / 52860108 | $98.04 | 59 min | 0 |

## Collection

- episodes: 289 (success 275, fail 12, unsure 2; 95%)
- unsure episodes deleted and redone: 4
- resets: 678 (ok 544, fail 134, manual 0)
- rounds: 3
- VLM judge: 678 calls, $22.90
- operator review queue: 12 pending

## Datasets and training

- dataset `bee_v1`: exit 0, episodes {'total': 108, 'success': 100, 'fail': 6, 'unsure': 2}
- dataset `probe`: exit 0, episodes {'total': 289, 'success': 275, 'fail': 12, 'unsure': 2}
- dataset `bee_v2`: exit 0, episodes {'total': 108, 'success': 100, 'fail': 6, 'unsure': 2}
- dataset `bee_v3`: exit 0, episodes {'total': 120, 'success': 112, 'fail': 6, 'unsure': 2}
- dataset `bee_v4`: exit 0, episodes {'total': 193, 'success': 182, 'fail': 9, 'unsure': 2}
- dataset `bee_v6`: exit 0, episodes {'total': 225, 'success': 211, 'fail': 12, 'unsure': 2}
- dataset `bee_v7`: exit 0, episodes {'total': 225, 'success': 211, 'fail': 12, 'unsure': 2}
- training run `lora_v1`: dataset bee_v1, exit 0, 459 min, `cd /home/r2r/robot_learning_final_pj && uv run rlpj vla train --run-dir /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee --dataset bee_v1 --exp-name lora_v1 --num-train-steps 15000 --batch-size 16 --overwrite`
- training run `lora_h30`: dataset bee_v2, exit 255, 20 min, `RLPJ_CONFIG=pi05_kinova_lora_h30 RLPJ_RECIPE=/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/programs/recipe_h30.py setsid nohup bash scripts/train.sh /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee bee_v2 lora_h30 --overwrite > /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/notes/train_lora_h30.out 2>&1 < /dev/null & sleep 600; grep -E "Progress on|Error|error" /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/notes/train_lora_h30.out | tail -5`
- training run `run_3`: dataset None, exit 0, 1 min, `bash /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/programs/wait_train.sh lora_h30`
- training run `run_4`: dataset None, exit 0, 19 min, `bash /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/programs/wait_train.sh lora_h30`
- training run `run_5`: dataset None, exit 0, 19 min, `bash /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/programs/wait_train.sh lora_h30`
- training run `run_6`: dataset None, exit 0, 19 min, `bash /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/programs/wait_train.sh lora_h30`
- training run `lora_v3`: dataset bee_v3, exit 255, 178 min, `rlpj vla train --run-dir /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee --dataset bee_v3 --exp-name lora_v3 --config pi05_kinova_lora_v3 --recipe /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/programs/recipe_v3.py`
- training run `full_v6`: dataset bee_v6, exit 1, 11 min, `rlpj vla train --run-dir /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee --dataset bee_v6 --exp-name full_v6 --config pi05_kinova_full --num-train-steps 30000 --batch-size 4 --pytorch-weight-path /home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v5/30000`

## Evaluations

| eval | checkpoint | episodes | success | rate | met | DAgger corrections (success/total) |
|---|---|---|---|---|---|---|
| 01_lora_v1_14999 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v1/14999` | 0 | 0 | 0% | no | 0/0 |
| 02_lora_v1_14999 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v1/14999` | 0 | 0 | 0% | no | 0/0 |
| 03_lora_v1_14999 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v1/14999` | 12 | 0 | 0% | no | 0/0 |
| 04_lora_v1_14999 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v1/14999` | 20 | 0 | 0% | no | 14/18 |
| 05_lora_h30_11999 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_h30/11999` | 20 | 0 | 0% | no | 13/16 |
| 06_lora_v3_4500 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/lora_v3/4500` | 2 | 0 | 0% | no | 1/2 |
| 07_full_v1_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000` | 0 | 0 | 0% | no | 0/0 |
| 08_full_v1_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/ckpt/full_v1/30000` | 20 | 0 | 0% | no | 5/17 |
| 09_full_v1_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/ckpt/full_v1/30000` | 0 | 0 | 0% | no | 0/0 |
| 10_full_v1_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000` | 0 | 0 | 0% | no | 0/0 |
| 11_full_v1_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/agents/learn/ckpt/full_v1/30000` | 1 | 0 | 0% | no | 1/1 |
| 12_full_v1_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v1/30000` | 20 | 0 | 0% | no | 13/13 |
| 13_full_v5_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v5/30000` | 20 | 7 | 35% | no | 4/6 |
| 14_full_v6_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v6/30000` | 20 | 2 | 10% | no | 8/10 |
| 15_full_v5_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v5/30000` | 10 | 2 | 20% | no | 2/2 |
| 16_full_v6_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v6/30000` | 20 | 2 | 10% | no | 10/12 |
| 17_full_v6_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v6/30000` | 20 | 3 | 15% | no | 5/7 |
| 18_full_v5_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v5/30000` | 20 | 13 | 65% | no | 2/2 |
| 19_full_v5_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v5/30000` | 20 | 12 | 60% | no | 2/3 |
| 20_full_v7_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v7/30000` | 1 | 0 | 0% | no | 0/1 |
| 21_full_v7_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v7/30000` | 0 | 0 | 0% | no | 0/0 |
| 22_full_v7_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v7/30000` | 20 | 16 | 80% | yes | 1/2 |
| 23_full_v8_30000 | `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v8/30000` | 20 | 11 | 55% | no | 4/5 |

DAgger corrections: 117 (success 85, fail 27, unsure 5)

deployable checkpoint: `/home/r2r/robot_learning_final_pj/runs/20261104_1510_bee/checkpoints/full_v7/30000`
