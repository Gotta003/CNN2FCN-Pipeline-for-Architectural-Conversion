# CNN2FCN-Pipeline-for-Architectural-Conversion
A multi-stage deep learning pipeline that compresses a frozen convolutional teacher model into a compact, flat-input student network using decoupled knowledge distillation (DKD), low-rank structural priors (HSR), and evolutionary architecture search.

A Python implementation that combines ENFORCE logit refinement with NAS, built for KWS on 40x50 spectrogram inputs

# Pipeline Overview
| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `stage0_teacher.py` | Train CNN on spectrogram dataset |
| 1 | `stage1_teacher.py` | Pre-train ENFORCE Refiner and produce normalized latent targets |
| 2 | `stage2_hint.py` | HSR hint distillation - Bridge Stage 1 with hybrid hint + Focal Loss |
| 3 | `stage3_dkd.py` | DKD fine-tuning - Bridge Stage 2 with TCKD+NCKD+nuclear-norm regression |
| 4 | `stage4_anchor.py` | Train 1M-param anchor `CompressedHSRBridge` |
| 5 | `stage5_nas.py` | Evolutionary NAS with turnament selection, crossover and mutation per budget |
| 6 | `stage6_eval.py` | Open-set threshold calibration, benchmark table and plots |

# Project Structure
```bash

```

## Future Improvements

## Useful Papers
- [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679) 
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [ProxylessNAS](https://arxiv.org/abs/1812.00332)
- [Towards Open Set Deep Network](https://arxiv.org/abs/1511.06233)
- [Energy OOD](https://arxiv.org/pdf/2010.03759)

