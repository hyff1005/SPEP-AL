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

The three supplementary scripts currently have a duplicated `.sh.sh` extension. Their exact archived names are retained in this README. The intended shorter names are `run_supplementary_experiments.sh`, `run_additional_controls.sh`, and `run_wt11_pareto.sh`.

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

These files provide dependency specifications rather than fully locked records of the environments used for the reported experiments. Exact package versions, model checkpoint versions, and hardware details are not fully captured.

## Current execution status

The supplied scripts retain assumptions from the original research environment. Some shell scripts contain machine-specific paths, environment names, and references to a directory layout that differs from the archive. Source imports also refer to a package name different from the archived implementation directory name. No Python packaging configuration is included.

The archive therefore should not be described as portable or ready for complete reproduction immediately after extraction. Renaming the scripts does not resolve these execution assumptions.

## Materials not included

The following materials were not found in this archive:

- Processed variant FASTA files and label CSV files.
- Precomputed sequence embedding tables and pretrained model weights.
- Predicted wild-type and mutant structures.
- Structural descriptor tables and structural quality control tables.
- A standalone structural feature extraction program.
- Complete per-round result tables and experiment logs.
- Supplementary result summarization programs.
- A complete set of statistical analysis and manuscript plotting programs.
- A standalone license file or Python packaging configuration.

The absence of these materials does not mean the main modeling modules are absent. It means this archive alone is insufficient to regenerate every reported result from the source workbook.

## Validation status

All nine Python source files passed static syntax parsing during inspection of this archive. Dependencies were not installed, module imports were not executed, and experiments were not run. Runtime correctness and numerical reproducibility have not been established by this inspection.

For a complete reproducibility release, the source code should be accompanied by the required feature preparation and analysis components, input artifact access information, recorded software versions, and a mapping from generated outputs to the manuscript tables and figures.

## Attribution

This project builds on EVOLVEpro:

Jiang K, Yan Z, Di Bernardo M, et al. Rapid in silico directed evolution by a protein language model with EVOLVEpro. Science 387, eadr6006 (2025). DOI: 10.1126/science.adr6006.

The retained preprocessing documentation contains descriptions of the upstream workflow. Its paths and examples do not necessarily correspond to this archive. Project-specific structural feature and acquisition additions should be distinguished from the original implementation when documenting a public release.

## License and release information

A license is not specified by a standalone file in this archive. Verify the applicable upstream code and data terms before redistribution, and state the terms covering project-specific changes. The public release should also identify the authors, repository or archive location, and release version.
