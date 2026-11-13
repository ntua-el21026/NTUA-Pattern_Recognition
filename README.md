# NTUA Pattern Recognition coursework

Analytical exercises and two audio-learning laboratory assignments from the NTUA School of Electrical and Computer Engineering Pattern Recognition course. The lab submissions credit **Michail-Athanasios Peppas (03121026)** and **Panagiotis Avgerinos (03121023)**. They are team implementations based on course material; individual implementation responsibilities have not been reconstructed from the imported history.

## Coursework index

| Work | Implementation and evidence |
| --- | --- |
| Spoken-digit recognition | [Lab 1 notebook](lab/ex1/patrec1_03121026_03121023.ipynb), [Python export](lab/ex1/patrec1_03121026_03121023.py), [submitted report](lab/ex1/docs/submission/patrec1_03121026_03121023_report.pdf): speech features, classical classifiers, GMM-HMMs and recurrent networks |
| Music genre and affect prediction | [Lab 2 notebook](lab/ex2/patrec2_03121026_03121023.ipynb), [Python export](lab/ex2/patrec2_03121026_03121023.py), [submitted report](lab/ex2/docs/submission/patrec2_03121026_03121023_report.pdf): LSTM/CNN/AST classification, regression, transfer and multitask comparisons |
| Analytical exercises | [Set 1](analytical_exercises/set_1/): HMM Viterbi training, online LMS classification, RNN experiments, original solution, plots and logs. Set 2 was optional under a different exam scheme. |
| Semester project | [Team 12 LLaMA-style 500M workflow](https://github.com/ntua-el21026/ntua-llama-500m-nanotron) and [companion artifacts](https://github.com/ntua-el21026/ntua-llama-500m-artifacts); topic context is in [project/README.md](project/README.md) |

[lab/README.md](lab/README.md) explains the experiments, data locations, evaluation qualifications and rerun prerequisites. [docs/](docs/) retains reference material and exam notes; [lab_material/README.md](lab_material/README.md) identifies the external course-support revisions.

## Inspecting and reproducing the labs

```bash
git clone https://github.com/ntua-el21026/ntua-pattern-recognition.git
cd ntua-pattern-recognition
```

Both submitted notebooks and their recorded outputs are already in Git. A reviewer does not need access to the original personal Colab link or an expired notebook-export URL. Open the repository notebooks for inspection; use a separate working copy for execution.

Lab 1 was prepared for Colab with personal Drive paths. Its preparatory speech corpus and 3,000-file Free Spoken Digit Dataset are bundled, but the path cells must be adapted in the working copy. Lab 2 targets Kaggle and requires externally obtained music data plus the course's `lab3` helper modules. The repository stores course-support Git commit references without a `.gitmodules` file; ordinary submodule initialization does not retrieve them. See the [support-code instructions](lab_material/README.md).

There is no consolidated environment lock or automated numerical test suite. The notebooks use NumPy, pandas, scikit-learn, librosa, matplotlib, PyTorch and tqdm; HMM work uses pomegranate, while the music work additionally uses SciPy, timm, kagglehub and course modules. Notebook setup cells contain installs and downloads. Review them against the managed runtime before running anything. The `.py` exports retain notebook syntax and are not installable libraries or ordinary command-line entry points.

## Experimental status

The reports and outputs are preserved coursework evidence, including weak results. Lab 1's reported 98.67% test accuracy belongs to its recorded experiment and selection procedure. Its final HMM code can retrain a digit after observing zero test accuracy, so that result should not be presented as an untouched-test estimate without that qualification. Music classification reports accuracy and class-wise metrics; affect prediction uses MSE and Spearman correlation. Splits and model-selection details differ between experiments.

Documentation validation inspected notebook structure, paths, helper references and evidence links. No training, numerical replay or new performance measurement was performed. Reproduction requires recovering compatible dependencies, data/model versions and the original protocol; preserve submitted notebooks and write fresh results separately.

## Authorship and reuse

The [root MIT license](LICENSE) covers original contributions within the contributors' rights. Course handouts, starter repositories, papers, books, tutorial notebooks, pretrained models and datasets have separate terms. The bundled [Free Spoken Digit Dataset notice](lab/ex1/material/free-spoken-digit-dataset/README.md#license) identifies **CC BY-SA 4.0**, which must be retained.

The laboratory assignment PDFs under `lab/ex1/docs/` and `lab/ex2/docs/` prohibit publishing solutions on GitHub or other websites. Analytical assignment statements also restrict posting solutions and identify teaching-team intellectual property. Those notices remain intact; an applicable documented permission is required for public distribution of restricted material. No permission is inferred from the root license or the repository's current visibility. Existing authorship, assistance disclosures and submitted evidence are preserved.
