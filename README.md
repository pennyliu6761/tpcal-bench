# tpcal-bench

Code for the manuscript *Small LLM Judges Detect Unseen Tool-Poisoning Attacks but Miscalibrate Across Tool
Ecosystems: A Cue-Controlled Benchmark and Tail-Aware Few-Shot Calibration* (Liu and Fan, submitted to *Applied Sciences*).

The repository contains the code and the cached detector scores. It does not contain the source datasets; they are
downloaded from their original repositories at pinned commits (step 00).

## Setup

```
conda create -n tpcal python=3.11 -y
conda activate tpcal
pip install -r requirements.txt
```

Steps 50 and 51 run the zero-shot detectors and need a GPU (tested on an 8 GB RTX 4060), the CUDA build of PyTorch and
`transformers` (see `requirements.txt`). All other steps run on a CPU in seconds to minutes. The models are public and
are downloaded anonymously from the Hugging Face Hub on first use; no account or token is needed.

## Reproduce the paper

Run from the repository folder. With the cached scores in `work/results/probe/`, only the last three steps are needed:

```
python scripts/52_probe_calibration.py
python scripts/60_paper_figures.py
python scripts/61_paper_tables.py
```

Full pipeline from the public data:

| Step | Script | Paper | Output |
|---|---|---|---|
| 00 | `00_fetch_data.py` | data | `external/` (four source repositories at pinned commits; no git needed) |
| 01 | `01_build_corpus.py` | Table 1 | `work/corpus/`, `work/results/tables/data_summary.csv` |
| 13 | `13_shortcut_audit.py` | Table 4 | `work/results/tables/audit_*.csv` |
| 50 | `50_semantic_probe.py` | first probe (optional) | `work/results/probe/probe_scores.csv` (TF-IDF LR reference, PI classifier, one judge) |
| 51 | `51_llm_judges.py` | Tables 2, 5 | `work/results/probe/judges_scores.csv`, `judges_summary.csv` (3 judges x 3 prompts, GPU, 1-2 h) |
| 52 | `52_probe_calibration.py` | Tables 6-8 | `work/results/probe/calib_*.csv` (all calibration rules, 100 draws) |
| 60 | `60_paper_figures.py` | Figures 1-8, Table 9 | `results/paper_figures/` |
| 61 | `61_paper_tables.py` | Tables 4-8, tests | `results/paper_tables/` |
| 62 | `62_compare_runs.py` | check | compares a fresh run of step 51 with the cached scores |

To check that the detectors reproduce, rerun them into a separate folder and compare (the cached scores are not touched):

```
set TPC_WORK=work_verify
python scripts/01_build_corpus.py
python scripts/51_llm_judges.py
set TPC_WORK=
python scripts/62_compare_runs.py --new work_verify/results/probe
```

GPU inference in half precision is not bit-exact, so differences in the third decimal of an AUROC are expected.

`python scripts/check_install.py` checks the code files, the checksums of the six data files and the installed
packages. Paths can be changed with the environment variables `TPC_DATA_ROOT` (data, default `external/`) and
`TPC_WORK` (intermediate results, default `work/`).

## Troubleshooting (Windows)

If an import fails with `DLL load failed ... blocked by application control policy`, Windows is blocking a compiled
module of pandas or scikit-learn. Open the Anaconda Prompt as administrator, or install versions that are not blocked
(`pip install "pandas==2.2.3" "scikit-learn==1.5.2"`). `python scripts/check_install.py` imports every compiled module
the scripts use and reports any that are blocked.

## Cached scores

`work/results/probe/judges_scores.csv` holds one row per description and one column per detector (judge score
log p(Yes) - log p(No), PI classifier logit, TF-IDF LR score) together with the ecosystem, label, attack family and
split. It contains no description text. `judges_summary.csv` holds the per-ecosystem detection metrics.
`work/results/tables/` holds the outputs of steps 01 and 13, which need the source data.

## Data sources

| Repository | Commit | Used for | Licence |
|---|---|---|---|
| [zhiqiangwang4/MCPTox-Benchmark](https://github.com/zhiqiangwang4/MCPTox-Benchmark) | f85189f | MCP-curated benign, MCPTox-style attacks | no licence file; used for research, not redistributed |
| [amaneja21/tp-detect](https://github.com/amaneja21/tp-detect) | 804a36b | MCP-wild benign, hand-written attacks | MIT |
| [HowieHwong/MetaTool](https://github.com/HowieHwong/MetaTool) | 35e81bb | Plugin benign | MIT |
| [ryoungj/ToolEmu](https://github.com/ryoungj/ToolEmu) | ac4a7ab | ToolEmu benign | Apache-2.0 |

If the automatic download in step 00 is blocked, download the zip of each commit in a browser
(`https://codeload.github.com/<owner>/<repo>/zip/<full commit>`, full hashes in `tpc/data.py`), unzip it into
`external/` and rename the folder to the repository name.

The poisoned descriptions are research attack samples. The code only scores them as text and never executes them.
