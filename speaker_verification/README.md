# Inclusive Speaker Verification and Fairness Analysis

**Team:** Sachin Patil (M24DE3067) · Venkatesh Doddi (M24DE3084) · Akash Badola (M24DE3008) · Ashish Gajbhiye (M24DE3018)

A SpeechBrain-based pipeline for equitable speaker recognition across demographic groups. This project extends standard speaker verification with **explicit fairness auditing, robustness testing, and explainability analysis**.

---

## Project Structure

```
speaker_verification/
├── config/
│   └── config.yaml          # All hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── utils.py             # Config, logging, seeding, checkpoints
│   ├── features.py          # Log-Mel Filterbank / MFCC extraction
│   ├── model.py             # X-Vector TDNN + ECAPA-TDNN
│   ├── dataset.py           # VoxCeleb + Verification datasets
│   ├── augment.py           # Noise augmentation (MUSAN + RIR)
│   ├── evaluate.py          # EER, minDCF, PLDA scoring
│   ├── fairness.py          # Per-group metrics, calibration, plots
│   ├── explain.py           # Integrated Gradients saliency
│   └── pretrained.py        # SpeechBrain pretrained wrapper
├── tests/
│   └── test_pipeline.py     # Pytest unit tests (all components)
├── notebooks/
│   └── demo.ipynb           # Interactive walkthrough (optional)
├── data/                    # Place datasets here (see below)
├── results/                 # Auto-created; all outputs saved here
├── main.py                  # Pipeline runner
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo (No Data Required)

The pipeline runs on **synthetic data** when real datasets are absent:

```bash
python main.py --phase demo
```

This trains a small model on synthetic utterances, runs augmentation, performs fairness analysis, and generates saliency maps. All outputs are saved to `results/`.

### 3. Run Individual Phases

```bash
python main.py --phase train      # Phase 2: train x-vector from scratch
python main.py --phase augment    # Phase 3: robustness under noise
python main.py --phase fairness   # Phase 4: demographic fairness analysis
python main.py --phase explain    # Phase 5: Integrated Gradients saliency
python main.py --phase all        # Run all phases sequentially
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Datasets (Optional – pipeline works without them)

| Dataset | Purpose | Source |
|---------|---------|--------|
| VoxCeleb1 | Training & evaluation | [robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) |
| VoxCeleb2 | Additional training data | Same link |
| MUSAN | Noise augmentation | [openslr.org/17](https://openslr.org/17) |
| RIR Noises | Reverberation | [openslr.org/28](https://openslr.org/28) |

After downloading, update paths in `config/config.yaml`.

**CSV Format** – Create `data/voxceleb1/train.csv`:

```csv
path,speaker_id,gender,nationality
data/voxceleb1/id10001/xyz/00001.wav,0,male,native
...
```

**Trials CSV** – Create `data/voxceleb1/trials.csv`:

```csv
path1,path2,label,gender1,gender2,nationality1,nationality2
data/.../utt1.wav,data/.../utt2.wav,1,male,male,native,native
...
```

---

## Pipeline Details

### Phase 2 – Feature Extraction & Model Training

- Log-Mel Filterbank (80 bins) or MFCC (40 coefficients)
- 5-layer TDNN x-vector model with statistics pooling
- ECAPA-TDNN alternative (channel-attention pooling)
- Cosine annealing LR schedule, gradient clipping

### Phase 3 – Robustness Testing

- MUSAN noise at SNR 5–20 dB
- Room Impulse Response convolution
- Speed perturbation (0.9×, 1.0×, 1.1×)
- Reports EER delta: clean vs degraded

### Phase 4 – Fairness Evaluation & Mitigation

- Per-group EER for male/female and native/non-native speakers
- 95% bootstrap confidence intervals (NIST practice)
- Fairness gap = max(EER) − min(EER) across groups
- **Mitigation:** Equalized-FPR threshold calibration
- **Mitigation:** Inverse-frequency sample reweighting

### Phase 5 – Explainability

- Integrated Gradients attribution (Sundararajan et al., 2017)
- Per-frequency attribution shows which Mel bins drive decisions
- Saliency maps saved as PNG for each trial pair
- Frequency importance summary across all samples

---

## Results Directory

```
results/
├── checkpoints/
│   └── best_xvector.pt
├── logs/
│   └── pipeline.log
├── fairness/
│   ├── fairness_before.csv
│   ├── fairness_after.csv
│   ├── fairness_eer_comparison.png
│   └── score_distributions.png
├── explainability/
│   ├── saliency_sample_0.png
│   ├── ...
│   └── frequency_importance.png
└── robustness_results.json
```

---

## Key References

1. Snyder et al., "X-Vectors: Robust DNN Embeddings for Speaker Recognition," ICASSP 2018.
2. Desplanques et al., "ECAPA-TDNN," Interspeech 2020.
3. Dehak et al., "Front-end Factor Analysis for Speaker Verification," IEEE TASLP 2011.
4. Ravanelli et al., "SpeechBrain: A PyTorch-based Speech Toolkit," Interspeech 2021.
5. Sundararajan et al., "Axiomatic Attribution for Deep Networks," ICML 2017.

---

## License

Academic use only. Datasets (VoxCeleb, MUSAN) are subject to their own licences.
