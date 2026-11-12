# Pattern_Recognition

Coursework and team lab assignments from the NTUA ECE Pattern Recognition course
(student id 03121026).

## Repository layout

```
Pattern_Recognition/
|-- analytical_exercises/   # homework sets, code, plots, results
|-- docs/                   # bibliography, exam notes, repo structure
|-- lab/                    # lab exercises (ex1, ex2) with reports and data
|-- lab_material/           # course-provided GitHub repos
`-- project/                # semester project topic list + selected topic
```

## analytical_exercises/

- set_1: assignment statement, solution PDF, and programs (HMM Viterbi training,
  LMS classification, RNN experiments, plotting scripts, logs/results).
- set_2: was optional with a different exam scheme.

## lab/

- docs/: report and notebook templates.
- ex1: notebook and Python script, reports (PDF/DOCX), submission bundle, figures,
  and the spoken digit dataset (wav files).
- ex2: notebook and Python script, reports (PDF/DOCX), submission bundle, helper
  scripts, codex experiments, figures, and a data loading tutorial.

## docs/

- bibliography: papers, notes, and course references (PDF/TXT/Notebook).
- exam_notes: LaTeX source and compiled PDF.
- project_structure: script and a captured tree of the repo.

## lab_material/

Reference material pulled from course GitHub repos (patrec-labs and prep-lab).
These include lab support code and prep notebooks.

## project/

Topic list for the semester project (PDF/DOCX) and `project/README.md` with the
selected topic (SPEC-1): https://github.com/ntua-el21026/NTUA-Llama_500M_Nanotron_Full

## License

See `LICENSE`.

## Authorship, status and limitations

The lab reports credit Michail-Athanasios Peppas (03121026) and Panagiotis Avgerinos (03121023), submitted on 13 November and 8 December 2025. These are team implementations based on course material. Per-person implementation responsibilities have not been reconstructed from the bulk-import history. The semester LLaMA project is a separate Team 12 artifact linked above.

Lab 1 implements spoken-digit HMM/RNN experiments; Lab 2 studies music genre and affect prediction with LSTM/CNN/AST, including regression, transfer and multitask comparisons. The preserved Lab 1 report's 98.67% test accuracy is specific to its task/split/model-selection protocol. Weak results and notebook outputs remain part of the evidence. No numerical replay or new performance claim accompanies portfolio preparation.

## Setup and reproduction

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ntua-el21026/NTUA-Pattern_Recognition.git
cd NTUA-Pattern_Recognition
```

Open `lab/ex1/patrec1_03121026_03121023.ipynb` or `lab/ex2/patrec2_03121026_03121023.ipynb`; the adjacent `.py` files are notebook exports and may contain Colab setup or notebook-specific assumptions. Inspect setup cells before execution rather than importing exports as a library. Exact resolved dependency versions are not frozen.

Lab 1 imports NumPy, pandas, scikit-learn, librosa, matplotlib, pomegranate, PyTorch and tqdm. Lab 2 additionally uses SciPy, timm, kagglehub and course-local dataset/model modules. HMM API versions, audio processing, model downloads and local paths require particular attention. Use the intended managed notebook environment and compatible provider numerical stack; a local PyTorch installation is unnecessary for source review. Record actual packages, hardware, seeds, source/data/model hashes and chronological train/validation/test decisions before a rerun. Keep fresh output separate from the submitted notebooks/reports.

## Conditional public release and third-party terms

This remains a conditional public portfolio candidate. Both laboratory assignment PDFs in `lab/ex1/docs/` and `lab/ex2/docs/` prohibit publishing laboratory solutions on GitHub or other websites. The analytical assignment statements also prohibit posting solutions on any platform and identify teaching-team intellectual property. Original notices, solutions and outputs remain intact. An applicable documented permission is required for a public release containing the restricted solutions. This preparation contacted no instructors and removed no coursework to bypass those terms.

The bundled Free Spoken Digit Dataset identifies CC BY-SA 4.0 in its own README. Retain that attribution/license and review applicable sharing obligations; root MIT is not its sole license. Course repositories, handouts, books, papers, pretrained models and tutorial notebooks need separate provenance/rights review. Root MIT applies to original contributions only within the contributors' rights. Public-first curation can use original explanatory writing and independently reusable source whose rights are established, while excluding restricted solutions, course prompts/starters and third-party datasets. The current local evidence copy supports that review without altering historical outputs.
