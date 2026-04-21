# ResNet18 Detection Pipeline Notes

This document complements the notebook workflow in [01_resnet18_voc_detection_detailed.ipynb](01_resnet18_voc_detection_detailed.ipynb) with practical guidance for understanding outputs and controlling memory use.

## Full detection pipeline for one image

At inference time, a single image goes through these stages:

1. Input tensor `(C, H, W)` is normalized and batched by `model.transform`.
2. ResNet18 + FPN creates multi-scale feature maps.
3. RPN proposes candidate regions likely to contain objects.
4. RoI Align crops fixed-size features for each proposal.
5. RoI heads produce:
   - class logits per proposal
   - box regression offsets per proposal
6. Softmax converts logits to class probabilities.
7. Score thresholding removes low-confidence predictions.
8. NMS suppresses duplicate overlapping boxes.
9. Final outputs are `boxes`, `labels`, `scores`.

### Logits value vs prediction value

- Logits are raw unbounded classifier outputs from the RoI head.
- Prediction scores are post-softmax probabilities in `[0, 1]`.
- The notebook displays final `scores` after filtering and NMS, so they are directly interpretable confidence values.

## Memory optimization guide (GPU + CPU)

### GPU memory controls

1. Reduce image size
- Smaller training resize (for example shorter side 480 instead of 800) can significantly reduce activation memory.
- Trade-off: lower spatial detail may hurt small-object detection.

2. Reduce batch size
- Detection models are memory-heavy. Batch size 1 or 2 is common.
- If memory is tight, prefer lower batch size before reducing model depth.

3. Use mixed precision (AMP)
- Use `torch.cuda.amp.autocast` and `GradScaler` for training.
- Typical gain: lower memory footprint and faster compute on modern GPUs.

4. Gradient accumulation
- Simulate larger effective batch size without storing more samples at once.
- Example: accumulate gradients for `k` steps before `optimizer.step()`.

5. Freeze more backbone layers
- Fewer trainable layers means fewer gradients stored.
- In this notebook, `trainable_layers` in `resnet_fpn_backbone` is a direct knob.

6. Keep post-processing limits reasonable
- Very high proposal counts increase memory and compute in RoI heads.
- If needed, tune proposal limits and score thresholding for lower overhead.

### CPU/RAM controls

1. Avoid full dataset caching in RAM unless required
- Keep lazy loading from disk for large datasets.

2. Tune DataLoader workers carefully
- Higher `num_workers` can speed loading but also increases memory use.
- Start low (`0` or `2`) and scale based on RAM availability.

3. Use `pin_memory=True` only when using GPU training
- Helps host-to-device transfer but slightly increases host memory usage.

4. Minimize unnecessary tensor copies
- Reuse tensors where possible and avoid repeated conversions between numpy and torch in loops.

## Recommended settings by hardware profile

### Low memory profile (GPU <= 6GB, RAM <= 16GB)

- batch size: `1`
- workers: `0-2`
- enable AMP
- freeze more backbone layers
- use conservative image resize

### Medium memory profile (GPU 8-12GB, RAM 16-32GB)

- batch size: `2`
- workers: `2-4`
- enable AMP
- keep moderate trainable backbone depth

### High memory profile (GPU >= 16GB, RAM >= 32GB)

- batch size: `2-4` (depending on image size)
- workers: `4-8`
- AMP still recommended for throughput
- can increase image resolution and training subset size

## OOM troubleshooting table

| Symptom | Likely cause | Quick fix |
|---|---|---|
| CUDA out of memory at forward pass | Activations too large from image size/batch | Reduce resize and batch size |
| OOM during backward pass | Too many trainable parameters and gradients | Freeze more layers; enable AMP |
| Training stalls with high RAM usage | Too many workers or cached samples | Reduce `num_workers`; avoid caching |
| Sudden memory spikes in inference | Too many proposals or low threshold keeps many boxes | Raise score threshold; reduce proposal limits |

## Practical checklist before longer training

1. Run one epoch with current settings and observe peak memory.
2. Confirm loss decreases without OOM.
3. Increase epochs only after memory is stable.
4. Save checkpoints every epoch to avoid losing progress.
