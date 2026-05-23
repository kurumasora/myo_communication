# Teacher Data Evaluation Summary

## Configuration

- epochs: 30
- batch_size: 20
- learning_rate: 0.001
- hidden_size: 50
- seed: 42

## Metrics

- train: accuracy=0.9471, loss=0.1739, samples=4480
- val: accuracy=0.9429, loss=0.1756, samples=560
- test: accuracy=0.9393, loss=0.2075, samples=560

## Test Classification Report

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| no_motion | 0.9281 | 0.9556 | 0.9416 | 135 |
| a | 0.9463 | 0.9338 | 0.9400 | 151 |
| i | 0.9225 | 0.9357 | 0.9291 | 140 |
| u | 0.9615 | 0.9328 | 0.9470 | 134 |

## Test Confusion Matrix

| actual/predicted | no_motion | a | i | u |
| --- | ---: | ---: | ---: | ---: |
| no_motion | 129 | 0 | 3 | 3 |
| a | 1 | 141 | 7 | 2 |
| i | 4 | 5 | 131 | 0 |
| u | 5 | 3 | 1 | 125 |
