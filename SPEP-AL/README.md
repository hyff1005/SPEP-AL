# SPEP-AL

SPEP-AL is a research implementation for retrospective protein activity learning that combines protein language model representations, structural descriptors, and positional entropy based batch acquisition. It builds on the EVOLVEpro framework.

This archive includes the main modeling code, benchmark preprocessing scripts, an ESM embedding extraction program, experiment configurations, and source data. It is a source code release snapshot; it does not include every intermediate artifact or analysis program required to reproduce all manuscript results.

## Archive layout

All paths below are relative to the outer `SPEP-AL/` directory obtained after extraction.

| Path | Description |
| --- | --- |
| `data/Source.xlsx` | Source benchmark workbook. |
| `data/wt_fasta/` | Twelve wild-type protein FASTA files. |
| `environment.yml` | Main modeling environment specification, named `SPEP-AL`. |
| `plm_environment.yml` | Protein language model environment specification, named `plm`. |
| `scripts/process/dms_process.py` | DMS benchmark preprocessing script. |
| `scripts/process/exp_process.py` | Experimental sequence processing examples. |
| `scripts/process/README.md` | Processing documentation inherited from the original workflow. |
| `scripts/plm/esm2_3B_dms.sh` | Shell script associated with ESM embedding extraction. |
| `scripts/dms/dms_main.py` | Command line entry point for modeling experiments. |
| `scripts/dms/esm2_3B.sh` | DMS experiment shell script. |
| `scripts/dms/run_supplementary_experiments.sh` | Supplementary experiment groups covering structural features, acquisition settings, and sensitivity analyses. |
| `scripts/dms/run_additional_controls.sh` | Additional control experiment configurations. |
| `scripts/dms/run_wt11_pareto.sh.sh` | Pareto acquisition configuration using eleven wild-type environment descriptors. |
| `SPEP-AL/plm/esm/extract.py` | ESM sequence embedding extraction implementation. |
| `SPEP-AL/src/data.py` | Data loading and feature preparation. |
| `SPEP-AL/src/evolve.py` | Iterative learning and experiment orchestration. |
| `SPEP-AL/src/model.py` | Regression models and acquisition methods. |
| `SPEP-AL/src/process.py` | Data and sequence processing utilities. |
| `SPEP-AL/src/utils.py` | Shared dimensionality reduction utilities. |

The inner `SPEP-AL/` directory is part of the supplied archive and contains the implementation modules. It is distinct from the outer archive root.


## Implementation overview

The project contains the following components:

- Source benchmark preprocessing and label preparation.
- Sequence representation extraction using ESM.
- Data loading and feature handling for sequence and structural representations.
- Iterative regression and batch acquisition.
- Supplementary experiment configurations and control comparisons.

The supplementary scripts describe separate experiment groups. They are not a single verified entry point covering every table and figure in the manuscript.

## Included data

The archive provides the source workbook and twelve wild-type FASTA files. Processed variant data and precomputed model inputs are not bundled.

The included workbook should be distinguished from experiment-ready label tables, variant sequence files, embeddings, and structural feature tables. These are different stages of data preparation and are not interchangeable.

## Environment specifications

`environment.yml` lists the main scientific Python dependencies, including NumPy, pandas, SciPy, scikit-learn, scikit-learn-extra, XGBoost, PyTorch, Biopython, openpyxl, Matplotlib, and Seaborn.

`plm_environment.yml` describes a separate environment for protein language models, including Python 3.10, PyTorch, fair-esm, and additional language model dependencies.


## License and release information

A license is not specified by a standalone file in this archive. Verify the applicable upstream code and data terms before redistribution, and state the terms covering project-specific changes. The public release should also identify the authors, repository or archive location, and release version.
