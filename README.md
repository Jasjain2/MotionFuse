# MotionFuse: Sensor Fusion for Human Movement Classification

## Overview

MotionFuse classifies athletic movements by fusing a wrist-worn inertial sensor (IMU) with body pose extracted from RGB video. It uses 8 sports-related actions from the UTD-MHAD dataset. Each trial is reduced to 76 IMU features and 396 pose features (from MediaPipe landmarks). The two are standardized, concatenated and classified with a Random Forest. The model is tested on 3 subjects it never saw during training. The fused model reaches **94.8% ± 1.9%** mean accuracy over 5 seeds, against **88.1% ± 1.4%** for pose only and **63.7% ± 2.5%** for IMU only.

## Key Results

Test accuracy on subjects 6-8 (96 trials). Mean ± sample std over 5 seeds (42-46); every model trains on subjects 1-4.

| Model                 | Features            | Accuracy (mean ± std) |
|-----------------------|---------------------|-----------------------|
| Random Forest         | IMU-only (76)       | 63.7% ± 2.5%          |
| Random Forest         | Pose-only (396)     | 88.1% ± 1.4%          |
| **Random Forest**     | **Fused (472)**     | **94.8% ± 1.9%**      |
| MLP (256-128, dp=0.3) | IMU-only (76)       | 66.2% ± 3.3%          |
| MLP (256-128, dp=0.3) | Pose-only (396)     | 85.6% ± 0.9%          |
| MLP (256-128, dp=0.3) | Fused (472)         | 89.6% ± 2.2%          |
| MLP (64, dp=0.5)      | Fused (472)         | 90.6% ± 2.7%          |
| SVM (RBF)             | Fused (472)         | 89.6% (deterministic) |

The production model (Random Forest, seed 42) scores 92.7% on the test set (95% bootstrap CI over test trials: 86.5% - 97.9%).

**Main fusion finding:** fusion is critical for **throw, catch and tennis_serve**. For these actions IMU-only and pose-only confuse the movement with different actions, so their errors are complementary.

## Dataset

[UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) (University of Texas at Dallas Multimodal Human Action Dataset) contains 27 actions performed by 8 subjects, with 4 repetitions each. It records RGB video, depth, skeleton and a wearable inertial sensor (3-axis accelerometer and 3-axis gyroscope at 50 Hz). The RGB videos are 640×480 at 15 fps.

MotionFuse uses the RGB and inertial modalities for 8 actions:

| ID | Action           | ID | Action          |
|----|------------------|----|-----------------|
| 5  | throw            | 14 | baseball_swing  |
| 7  | basketball_shoot | 15 | tennis_forehand |
| 12 | bowling          | 17 | tennis_serve    |
| 13 | boxing           | 20 | catch           |

These 8 are the dataset's sports movements (throwing, shooting, swinging, striking and catching), chosen for athletic relevance. The other actions, such as gestures, shape drawing and locomotion, are left out. That gives 8 actions × 8 subjects × 4 trials = 256 trials, all of them present.

> C. Chen, R. Jafari, N. Kehtarnavaz. "UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor." IEEE ICIP, 2015.

## Method

```
 wrist IMU (.mat, 50 Hz)                       RGB video (.avi, 15 fps)
 [frames, 6]: accel xyz, gyro xyz              640x480 frames
          |                                            |
          v                                            v
 IMU feature extraction                  MediaPipe PoseLandmarker (heavy)
 stats + FFT peaks + magnitudes          33 landmarks x (x, y, z) per frame
          |                              gap filling (linear interpolation)
          |                                            |
          |                                            v
          |                              Normalization: hip-centred,
          |                              torso-length scaled
          |                                            |
          |                                            v
          |                              Pose features: position & velocity
          |                              mean/std
          v                                            v
     76 features                                 396 features
          |                                            |
          +-------------> StandardScaler <-------------+
                         (per modality, fit on train)
                                    |
                                    v
                        concatenate: 472 features
                                    |
                                    v
                      Random Forest (100 trees) -> 8 actions
```

**IMU feature extraction** ([src/imu_features.py](src/imu_features.py)). Each of the 6 channels contributes 7 statistics (mean, std, min, max, median, RMS, zero-crossing rate) and the magnitudes of its 5 largest FFT peaks. FFT magnitudes are divided by trial length so trials of different duration are comparable. Four cross-channel features (mean and std of the acceleration and gyroscope magnitude vectors) bring the total to 76.

**Pose extraction and normalization** ([src/pose_extraction.py](src/pose_extraction.py), [src/normalization.py](src/normalization.py)). MediaPipe's PoseLandmarker (heavy model, video mode with tracking) gives 33 landmarks per frame. Frames where detection fails are filled by linear interpolation, and raw landmarks are cached to `data/pose_cache/`. Each frame is then translated so the mid-hip is the origin and divided by torso length (mid-hip to mid-shoulder). This makes the pose independent of where the person stands and how far they are from the camera; y is first rescaled by the 4:3 aspect ratio so distances are geometrically correct. The features are the mean and std of every landmark coordinate's position and frame-to-frame velocity: 33 × 3 × 4 = 396.

**Feature fusion** ([src/fusion.py](src/fusion.py)). Each modality gets its own `StandardScaler`, fit on the training subjects only. Without this, gyroscope values in deg/s would dwarf the normalized pose coordinates. The scaled blocks are concatenated into a 472-dimensional vector.

**Classification** ([src/evaluate.py](src/evaluate.py)). A scikit-learn Random Forest with 100 trees is the production model. PyTorch MLPs ([src/models.py](src/models.py), [src/train.py](src/train.py)) are trained as comparisons, using Adam, cross-entropy loss and early stopping on validation loss. With only 128 training trials, every MLP architecture tried, down to a single linear layer, stayed below the Random Forest.

## Evaluation Protocol

- **Subject-wise split:** train on subjects 1-4 (128 trials), validate on subject 5 (32 trials, used only for MLP early stopping), test on subjects 6-8 (96 trials). No subject appears in more than one split, and scalers are fit on training subjects only.
- **Analyses** ([scripts/generate_report.py](scripts/generate_report.py)):
  - **Per-action:** accuracy, precision, recall and F1, plus confusion matrices for each modality.
  - **Per-subject generalization:** which actions fail for which test subject.
  - **Uncertainty:** Random Forest vote-fraction confidence, a reliability diagram with ECE, and predictive entropy.
  - **Failure cases:** every misclassified trial, grouped by confusion pattern.
  - **Modality contribution:** per-action accuracy of IMU-only vs pose-only vs fused.
- **5-seed averaging:** headline and modality-contribution numbers are means over seeds 42-46. The production model uses a fixed seed (42) rather than the best test seed, to avoid selecting on the test set.

## Key Findings

- **Fusion gains** (per-action accuracy, mean of 5 Random Forest seeds, over the best single modality):

  | Action       | IMU-only | Pose-only | Fused | Gain       |
  |--------------|----------|-----------|-------|------------|
  | throw        | 15.0%    | 55.0%     | 76.7% | +21.7 pts  |
  | catch        | 25.0%    | 80.0%     | 95.0% | +15.0 pts  |
  | tennis_serve | 50.0%    | 81.7%     | 96.7% | +15.0 pts  |

  Pose alone is enough for bowling, basketball_shoot and baseball_swing, and either modality alone handles boxing.
- **Errors concentrate in one subject and are low-confidence.** The production model misclassifies 7 of 96 test trials, and subject 7 accounts for 4 of them (tennis_forehand predicted as bowling twice, catch predicted as throw twice). No error has a confidence above 0.51, and in every error the true action is the runner-up. Throw and catch, which are mirror-image movements, account for 4 of the 7 errors.
- **The modalities make complementary errors.** Pooled over seeds, IMU-only most often confuses throw with basketball_shoot, while pose-only confuses it with catch. For catch the confusions are boxing (IMU) versus throw (pose), and for tennis_serve, boxing versus tennis_forehand. My interpretation, which the experiment does not isolate: the wrist IMU captures arm dynamics but misses body configuration, while pose captures whole-body shape but sees fast wrist motion only coarsely at 15 fps.

## Reproduction

Tested on Windows 11, Python 3.14, CPU only.

1. Clone the repository:
   ```bash
   git clone https://github.com/Jasjain2/MotionFuse.git motionfuse && cd motionfuse
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and organize UTD-MHAD (~1.1 GB, mostly the RGB archive). The download resumes if interrupted, and modalities already present are skipped:
   ```bash
   python setup_data.py
   ```
4. Extract poses. The first run takes about 25 min on CPU (24.5 min measured); later runs load from `data/pose_cache/` in seconds:
   ```bash
   python scripts/run_pose_extraction.py
   ```
5. Train and compare baselines:
   ```bash
   python scripts/run_baselines.py
   ```
6. Generate the full evaluation report:
   ```bash
   python scripts/generate_report.py
   ```
7. Results are written to `results/` (`summary.txt`, CSVs, figures, and models in `results/models/`).

`python scripts/run_pipeline.py` runs steps 4-6 in one go. `python scripts/run_mlp_tuning.py --all-variants` reproduces the MLP architecture and training-data experiments. Rerunning from a clean `results/` gives byte-identical CSVs and summary.

**MediaPipe note:** recent MediaPipe releases (including every wheel for Python ≥ 3.13) no longer ship the legacy `mp.solutions.pose` API. The code uses it when available and otherwise falls back to the Tasks `PoseLandmarker`. The fallback downloads the heavy model to `data/models/` on first use.

## Repository Structure

```
motionfuse/
├── README.md
├── requirements.txt
├── setup_data.py              Download (with resume), extract and verify UTD-MHAD
├── src/
│   ├── config.py              All paths, dataset selection, hyperparameters
│   ├── data_loader.py         Load .mat IMU and .avi video per trial; train/test split
│   ├── imu_features.py        76 IMU features per trial
│   ├── pose_extraction.py     MediaPipe pose per frame, gap filling, on-disk cache
│   ├── normalization.py       Hip-centred, torso-scaled pose; 396 pose features
│   ├── fusion.py              Per-modality standardization, concatenation, feature loading
│   ├── models.py              PyTorch MovementClassifier and named architecture variants
│   ├── train.py               Training loop with early stopping or fixed epochs
│   └── evaluate.py            Metrics, uncertainty, failure cases, modality analysis, plots
├── scripts/
│   ├── run_pose_extraction.py Pose extraction for all trials + detection summary
│   ├── run_baselines.py       IMU-only / pose-only / fused MLP, RF and SVM comparison
│   ├── run_mlp_tuning.py      MLP capacity and training-subject experiments
│   ├── generate_report.py     Full evaluation; writes every output to results/
│   └── run_pipeline.py        Pose extraction -> baselines -> report
├── data/                      (generated) dataset, pose cache, MediaPipe model
└── results/                   (generated) tables, figures, summary, trained models
```

## Limitations

- **Small dataset.** The data covers 8 subjects, with 128 training and 96 test trials. One test trial is worth 1.0 point of accuracy, and the 95% CI on the production model spans 11 points. Each subject-action cell has 4 trials (25-point steps), so per-action and per-subject differences are indicative, not significant.
- **No temporal modeling.** Features are whole-trial summaries, so the order of motion phases (wind-up, strike, follow-through) is discarded.
- **Offline pipeline.** Trials are pre-segmented and processed whole. MediaPipe heavy takes about 4-8 s per trial on CPU, there is no streaming segmentation, and IMU and video are aligned only by trial boundaries, not timestamps.
- **Random Forest confidence is not calibrated.** Confidence is the fraction of trees voting for the prediction, not a probability. The model is under-confident: ECE is 0.292, with mean confidence 0.64 against accuracy 0.93. It ranks trials usefully but shouldn't be read as a probability, and it isn't comparable to a neural network's softmax.
- **Model selection caveat.** The validation subject (s5) is near-saturated (97-99% for all MLPs), so it cannot rank models. The test set has also been viewed across several rounds of comparison, so treat the headline numbers as slightly optimistic. Leave-one-subject-out cross-validation on s1-5 would be a sounder selection protocol.
- **Lab conditions only.** The data comes from a single fixed camera, a plain background and one right-wrist IMU. MediaPipe z is relative depth, not metric. Generalization to field footage or other sensor placements is untested.

## Future Work

- **Temporal model:** a 1D CNN (or GRU) on the raw IMU and landmark sequences, to use motion-phase order.
- **More subjects:** to tighten confidence intervals and support leave-one-subject-out model selection.
- **Real-time inference endpoint:** streaming segmentation, a lighter pose model, and timestamp-level IMU/video sync.
- **Confidence calibration:** isotonic or Platt scaling fit by cross-validation on training subjects, checked against the reliability diagram.
- **Field-condition testing:** different cameras, backgrounds, occlusion and sensor placements.
