# Log Summary

The staged seed logs do not contain a completed `quantized_ttt_phased` result.
All three runs were interrupted during TTT compile/eval, so the last completed
validation metric is `quantized_sliding_window val_bpb`.

## Final Metrics

| Seed | Train stop step | Last scheduled val step | Final completed val_bpb | Artifact size |
| --- | ---: | ---: | ---: | ---: |
| 42 | 8597 | 8597 | 1.08934733 | 15,999,684 bytes |
| 314 | 8631 | 8631 | 1.09035192 | 15,997,730 bytes |
| 999 | 8620 | 8620 | 1.09285937 | 15,998,747 bytes |

## Mean

- `quantized_sliding_window val_bpb` mean: `1.09085287`

## Source Logs

- `logs/seed_42.log`
- `logs/seed_314.log`
- `logs/seed_999.log`
