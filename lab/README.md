# Audio-learning laboratory work

The notebooks are the primary executable narratives; their Python exports and PDF reports retain the original submissions. The two-person team is Michail Athanasios Peppas (03121026) and Panagiotis Avgerinos (03121023). Templates are in [docs/](docs/).

## Lab 1: speech features and spoken digits

Start with [the notebook](ex1/patrec1_03121026_03121023.ipynb) and [submitted report](ex1/docs/submission/patrec1_03121026_03121023_report.pdf). The work covers:

- Praat pitch/formant inspection recorded in the report, followed by MFCC, delta/delta-delta and summary-feature analysis.
- Gaussian Bayes, SVM, k-NN, logistic-regression and decision-tree comparisons.
- A recurrent sine-to-cosine exercise and spoken-digit GMM-HMM/recurrent-network experiments.

The committed notebook has 46 cells, including 20 code cells. These counts describe the preserved file, not a successful rerun.

### Data and path setup

Both speech resources are present locally:

| Resource | Repository path | Contents |
| --- | --- | --- |
| Preparatory corpus | [ex1/material/pr_lab2_data/](ex1/material/pr_lab2_data/) | 133 isolated-digit WAVs in `digits/`, plus `onetwothree1.wav` and `onetwothree8.wav` |
| Free Spoken Digit Dataset | [ex1/material/free-spoken-digit-dataset/](ex1/material/free-spoken-digit-dataset/) | 3,000 WAVs in `recordings/`, with upstream README and utilities |

In a **copy** of the notebook, replace the Google Drive mount/setup cell with paths appropriate to your checkout. With the kernel working directory at the repository root, the initial preparatory path is:

```python
from pathlib import Path
repository = Path.cwd().resolve()
dataset_path = str(repository / "lab/ex1/material/pr_lab2_data")
```

The later FSDD cell assigns a separate old Drive-based `dataset_path` and changes the working directory. Replace that assignment in the working copy with `str(repository / "lab/ex1/material/free-spoken-digit-dataset/recordings")`. Keep `repository` absolute, and inspect the subsequent directory changes rather than replacing only the first path. The HMM/recurrent sections expect `recordings/*.wav`; the final section also searches several relative locations.

For Colab, open/upload a notebook copy and make the repository data available in that runtime. The personal Drive layout embedded in the submission is historical, not a required storage account. NumPy/audio-processing and pomegranate API compatibility still need qualification; the notebooks do not pin resolved package versions.

### Evaluation interpretation

The preparatory classifier comparison uses a stratified 70/30 split with random state 0. FSDD instead fixes utterance indices 0–4 as test and derives validation data from the remaining pool. Preserve these distinct protocols and fit preprocessing only on the corresponding training data.

The final HMM section includes a test-triggered retry for a digit with zero test accuracy. Report that selection behavior alongside the retained 98.67% report result; do not equate it with a fresh, untouched holdout evaluation. Preserve the notebook's weaker configurations and outputs when comparing models.

## Lab 2: music genre and affect

Read [the notebook](ex2/patrec2_03121026_03121023.ipynb), [submitted report](ex2/docs/submission/patrec2_03121026_03121023_report.pdf) and [data-loading tutorial](ex2/material/lab2-data-loading-tutorial.ipynb). The notebook has 41 cells, including 21 code cells.

Steps 0–7 explore mel/chroma and beat-synchronized features, label consolidation, an LSTM classifier, CNNs and an Audio Spectrogram Transformer. Steps 8–10 compare single-target affect regression, classification-to-regression transfer and joint valence/energy/danceability prediction. The optional Step 11 is not part of the submission. Genre evaluation includes accuracy, precision, recall, F1 and confusion matrices; regression uses MSE and Spearman correlation.

### External inputs

The music arrays and trained checkpoints are not included in this repository. The notebook references:

- [geoparslp/patreco3-multitask-affective-music](https://www.kaggle.com/datasets/geoparslp/patreco3-multitask-affective-music): prepared music arrays and labels.
- [multitask-affective-music-lab-2022](https://www.kaggle.com/competitions/multitask-affective-music-lab-2022): competition source, subject to its access/participation requirements.
- [michaelpeppas/preparation-dataset](https://www.kaggle.com/datasets/michaelpeppas/preparation-dataset): an additional source referenced by the exported setup.
- [slp-ntua/patrec-labs](https://github.com/slp-ntua/patrec-labs): `dataset.py`, `lstm.py`, `modules.py` and other course helpers; see [the retained revision](../lab_material/README.md).

Kaggle is the original execution layout. Attach the required data there and confirm that the path `../input/patreco3-multitask-affective-music/data/` resolves from the working directory. Expected subdirectories include `fma_genre_spectrograms/`, `fma_genre_spectrograms_beat/` and `multitask_dataset/`, with train/test arrays and the corresponding label files. Later cells also contain absolute `/kaggle/input/` paths. The exported Colab setup uses `kagglehub`, but later path cells do not automatically use its returned download locations; adapt all of them in a separate notebook copy for another environment.

The notebook clones the course repository and adds its `lab3` directory to `sys.path`. Recover the [recorded support-code revision](../lab_material/README.md) before checking compatibility. The files [step8_regression.py](ex2/codex/step8_regression.py), [step9_transfer.py](ex2/codex/step9_transfer.py) and [step10_multitask.py](ex2/codex/step10_multitask.py) depend on preceding notebook globals/backbones and are supplementary experiments, not standalone CLI programs.

## Evidence and rerun record

Original figures and submission files remain under each exercise's `docs/`. The old Colab link and signed Kaggle export URL in the Python headers are historical source locations. The committed notebooks provide review access without either link.

Before a rerun, record the source commit, exact packages and managed runtime, hardware, seeds, data/helper/pretrained-model revisions and hashes, split definitions and checkpoint-selection rule. Store new checkpoints/results separately and label them as new observations.
