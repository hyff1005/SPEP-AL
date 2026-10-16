import os
from typing import Dict, List, Optional, Tuple, Union

from Bio import SeqIO
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

STRUCT_FEATURE_COLUMNS = [
    "wt_site_relative_sasa",
    "wt_contact_count_8A",
    "wt_local_mean_relative_sasa",
    "wt_local_mean_ca_distance",
    "wt_local_std_ca_distance",
    "wt_local_radius_of_gyration",
    "wt_site_to_centroid_normalized",
    "wt_sin_phi",
    "wt_cos_phi",
    "wt_sin_psi",
    "wt_cos_psi",
    "delta_site_relative_sasa",
    "delta_local_mean_relative_sasa",
    "delta_contact_count_8A",
    "mutation_site_ca_displacement",
    "local_ca_rmsd",
    "local_distance_rmsd",
    "local_max_ca_displacement",
    "delta_local_radius_of_gyration",
]


STRUCT_DELTA_COLUMNS = [
    "delta_site_relative_sasa",
    "delta_local_mean_relative_sasa",
    "delta_contact_count_8A",
    "mutation_site_ca_displacement",
    "local_ca_rmsd",
    "local_distance_rmsd",
    "local_max_ca_displacement",
    "delta_local_radius_of_gyration",
]


STRUCT_QC_INVALID_ALL_FLAGS = {
    "failed",
    "worker_failed",
    "invalid_variant_name",
}


STRUCT_QC_INVALID_DELTA_FLAGS = {
    "very_low_local_plddt",
    "high_alignment_rmsd",
}


STRUCT_QC_MIN_LOCAL_PLDDT = 50.0
STRUCT_QC_MAX_ALIGNMENT_RMSD = 5.0


STRUCT_CLIP_LOWER_QUANTILE = 0.01
STRUCT_CLIP_UPPER_QUANTILE = 0.99




def _tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _clean_variant_index(index: pd.Index) -> pd.Index:
    return pd.Index([str(value).strip() for value in index])


def _deduplicate_index(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.index.duplicated().any():
        duplicate_count = int(df.index.duplicated().sum())
        print(
            f"Warning: {source_name} contains {duplicate_count} duplicated variants. "
            f"Keeping first occurrence."
        )
        df = df[~df.index.duplicated(keep="first")]
    return df


def _quality_flag_tokens(value) -> set:
    if value is None or pd.isna(value):
        return set()

    text = str(value).strip().lower()
    if not text:
        return set()

    return {
        token.strip()
        for token in text.split(";")
        if token.strip()
    }


def _select_structural_schema(raw_struct: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    available_new_columns = [
        column for column in STRUCT_FEATURE_COLUMNS if column in raw_struct.columns
    ]

    if available_new_columns:
        missing_columns = [
            column for column in STRUCT_FEATURE_COLUMNS if column not in raw_struct.columns
        ]
        selected = raw_struct.copy()
        for column in missing_columns:
            selected[column] = np.nan

        if missing_columns:
            print(
                "Warning: structural file is missing "
                f"{len(missing_columns)} expected 3D feature columns: "
                f"{missing_columns}. They will be imputed."
            )

        return selected[STRUCT_FEATURE_COLUMNS].copy(), True

    print(
        "Warning: the new 19-dimensional structural schema was not detected. "
        "Falling back to all columns from the legacy structural file."
    )
    return raw_struct.copy(), False


def _apply_structural_qc(
    struct_numeric: pd.DataFrame,
    raw_qc: Optional[pd.DataFrame],
    is_new_schema: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {
        "qc_rows_available": 0,
        "qc_rows_missing": len(struct_numeric),
        "delta_invalidated": 0,
        "all_invalidated": 0,
    }

    if raw_qc is None or raw_qc.empty:
        return struct_numeric, stats

    qc = raw_qc.copy()
    qc.index = _clean_variant_index(qc.index)
    qc = _deduplicate_index(qc, source_name="structural QC")
    qc = qc.reindex(struct_numeric.index)

    qc_row_available = qc.notna().any(axis=1)
    stats["qc_rows_available"] = int(qc_row_available.sum())
    stats["qc_rows_missing"] = int((~qc_row_available).sum())

    quality_flags = (
        qc["quality_flag"]
        if "quality_flag" in qc.columns
        else pd.Series(index=qc.index, dtype=object)
    )
    flag_tokens = quality_flags.apply(_quality_flag_tokens)

    invalid_all_mask = flag_tokens.apply(
        lambda tokens: bool(tokens & STRUCT_QC_INVALID_ALL_FLAGS)
    )
    invalid_delta_mask = flag_tokens.apply(
        lambda tokens: bool(tokens & STRUCT_QC_INVALID_DELTA_FLAGS)
    )
    if "alignment_rmsd" in qc.columns:
        alignment_rmsd = pd.to_numeric(qc["alignment_rmsd"], errors="coerce")
        invalid_delta_mask = invalid_delta_mask | (
            alignment_rmsd > STRUCT_QC_MAX_ALIGNMENT_RMSD
        )

    local_plddt_columns = [
        column
        for column in ["wt_local_mean_plddt", "mut_local_mean_plddt"]
        if column in qc.columns
    ]
    for column in local_plddt_columns:
        local_plddt = pd.to_numeric(qc[column], errors="coerce")
        invalid_delta_mask = invalid_delta_mask | (
            local_plddt < STRUCT_QC_MIN_LOCAL_PLDDT
        )

    invalid_all_mask = invalid_all_mask.fillna(False)
    invalid_delta_mask = invalid_delta_mask.fillna(False) & (~invalid_all_mask)

    if invalid_all_mask.any():
        struct_numeric.loc[invalid_all_mask, :] = np.nan

    if is_new_schema and invalid_delta_mask.any():
        valid_delta_columns = [
            column
            for column in STRUCT_DELTA_COLUMNS
            if column in struct_numeric.columns
        ]
        struct_numeric.loc[invalid_delta_mask, valid_delta_columns] = np.nan

    stats["delta_invalidated"] = int(invalid_delta_mask.sum())
    stats["all_invalidated"] = int(invalid_all_mask.sum())
    return struct_numeric, stats


def _winsorize_struct_features(
    struct_numeric: pd.DataFrame,
    lower_quantile: float = STRUCT_CLIP_LOWER_QUANTILE,
    upper_quantile: float = STRUCT_CLIP_UPPER_QUANTILE,
) -> pd.DataFrame:
    clipped = struct_numeric.copy()

    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError(
            "Structural clipping quantiles must satisfy "
            "0 <= lower < upper <= 1."
        )

    for column in clipped.columns:
        values = pd.to_numeric(clipped[column], errors="coerce")
        finite_values = values[np.isfinite(values)]

        if finite_values.empty:
            clipped[column] = values
            continue

        lower = finite_values.quantile(lower_quantile)
        upper = finite_values.quantile(upper_quantile)

        if not (np.isfinite(lower) and np.isfinite(upper)):
            clipped[column] = values
            continue

        if lower < upper:
            clipped[column] = values.clip(lower=lower, upper=upper)
        else:
            clipped[column] = values

    return clipped


def _prepare_struct_features(
    raw_struct: pd.DataFrame,
    target_variants: List[str],
    raw_qc: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    if raw_struct is None or raw_struct.empty:
        return None

    target_variants = [str(variant).strip() for variant in target_variants]

    struct = raw_struct.copy()
    struct.index = _clean_variant_index(struct.index)
    struct = _deduplicate_index(struct, source_name="structural features")

    original_index = struct.index.copy()
    struct, is_new_schema = _select_structural_schema(struct)

    struct_aligned = struct.reindex(target_variants)

    struct_numeric = struct_aligned.apply(pd.to_numeric, errors="coerce")
    struct_numeric = struct_numeric.replace([np.inf, -np.inf], np.nan)

    missing_before_qc = int(struct_numeric.isna().sum().sum())

    struct_numeric, qc_stats = _apply_structural_qc(
        struct_numeric=struct_numeric,
        raw_qc=raw_qc,
        is_new_schema=is_new_schema,
    )

    missing_after_qc = int(struct_numeric.isna().sum().sum())
    qc_invalidated_values = max(0, missing_after_qc - missing_before_qc)

    struct_clipped = _winsorize_struct_features(struct_numeric)

    medians = struct_clipped.median(axis=0, skipna=True).fillna(0.0)
    struct_imputed = struct_clipped.fillna(medians)

    if struct_imputed.shape[1] > 0:
        scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
        )
        scaled_values = scaler.fit_transform(struct_imputed)
        struct_scaled = pd.DataFrame(
            scaled_values,
            index=struct_imputed.index,
            columns=struct_imputed.columns,
        )
    else:
        struct_scaled = pd.DataFrame(index=struct_imputed.index)

    struct_scaled = struct_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    total_variants = len(target_variants)
    variants_with_struct_row = int(
        pd.Index(target_variants).isin(original_index).sum()
    )
    total_imputed_values = int(struct_clipped.isna().sum().sum())

    schema_name = "new_19d" if is_new_schema else "legacy"
    print(
        "Successfully prepared structural features. "
        f"Schema: {schema_name}, "
        f"base dims: {struct_numeric.shape[1]}, "
        f"final dims: {struct_scaled.shape[1]}, "
        f"variants with structural row: {variants_with_struct_row}/{total_variants}, "
        f"QC rows available: {qc_stats['qc_rows_available']}/{total_variants}, "
        f"QC delta-invalidated variants: {qc_stats['delta_invalidated']}, "
        f"QC all-invalidated variants: {qc_stats['all_invalidated']}, "
        f"QC-invalidated values: {qc_invalidated_values}, "
        f"values imputed: {total_imputed_values}"
    )

    return struct_scaled



def load_dms_data(
    dataset_name: str,
    model_name: str,
    embeddings_path: str,
    labels_path: str,
    embeddings_file_type: str,
    embeddings_type_pt: str = "both",
    struct_path: str = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    labels_file = os.path.join(labels_path, f"{dataset_name}_labels.csv")
    embeddings_file = os.path.join(
        embeddings_path,
        f"{dataset_name}_{model_name}.{embeddings_file_type}",
    )

    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found at {labels_file}")
        return None, None, None

    labels = pd.read_csv(labels_file)

    if "variant" not in labels.columns:
        print("Error: 'variant' column not found in labels file.")
        return None, None, None

    if "activity" not in labels.columns:
        print("Error: 'activity' column not found in labels file.")
        return None, None, None

    labels = labels[labels["activity"].notna()].copy()
    labels["variant"] = labels["variant"].astype(str).str.strip()

    if labels["variant"].duplicated().any():
        duplicate_count = int(labels["variant"].duplicated().sum())
        print(
            f"Warning: labels contain {duplicate_count} duplicated variants. "
            f"Keeping first occurrence."
        )
        labels = labels.drop_duplicates(
            subset=["variant"],
            keep="first",
        ).copy()

    if embeddings_file_type == "csv":
        if not os.path.exists(embeddings_file):
            print(f"Error: Embeddings file not found at {embeddings_file}")
            return None, None, None
        plm_embeddings = pd.read_csv(embeddings_file, index_col=0)

    elif embeddings_file_type == "pt":
        if not os.path.exists(embeddings_file):
            print(f"Error: Embeddings file not found at {embeddings_file}")
            return None, None, None

        embeddings_dict = torch.load(embeddings_file, map_location="cpu")
        processed_dict = process_pt_embeddings(
            embeddings_dict,
            embeddings_type_pt,
        )
        if processed_dict is None:
            return None, None, None
        plm_embeddings = pd.DataFrame.from_dict(
            processed_dict,
            orient="index",
        )

    else:
        print("Invalid file type. Choose 'csv' or 'pt'")
        return None, None, None

    plm_embeddings.index = _clean_variant_index(plm_embeddings.index)
    plm_embeddings = _deduplicate_index(
        plm_embeddings,
        source_name="embeddings",
    )

    plm_embeddings = plm_embeddings.apply(pd.to_numeric, errors="coerce")
    plm_embeddings = plm_embeddings.replace([np.inf, -np.inf], np.nan)
    plm_medians = (
        plm_embeddings.median(axis=0, skipna=True).fillna(0.0)
    )
    plm_embeddings = plm_embeddings.fillna(plm_medians)

    common_variants = plm_embeddings.index.intersection(labels["variant"])
    if common_variants.empty:
        print(
            "Error: No common variants found between embeddings and labels "
            f"for {dataset_name}"
        )
        return None, None, None

    labels = labels[labels["variant"].isin(common_variants)].copy()
    labels = labels.sort_values(by=["variant"]).reset_index(drop=True)
    target_variants = labels["variant"].tolist()

    plm_embeddings = plm_embeddings.loc[target_variants]

    struct_df = None
    if struct_path is not None:
        struct_file = os.path.join(
            struct_path,
            f"{dataset_name}_struct.csv",
        )
        struct_qc_file = os.path.join(
            struct_path,
            f"{dataset_name}_struct_qc.csv",
        )

        if os.path.exists(struct_file):
            raw_struct = pd.read_csv(struct_file, index_col=0)

            raw_qc = None
            if os.path.exists(struct_qc_file):
                raw_qc = pd.read_csv(struct_qc_file, index_col=0)
            else:
                print(
                    f"Warning: Structural QC file {struct_qc_file} not found. "
                    "Structural features will be prepared without QC masking."
                )

            struct_df = _prepare_struct_features(
                raw_struct=raw_struct,
                target_variants=target_variants,
                raw_qc=raw_qc,
            )
        else:
            print(
                f"Warning: Structural file {struct_file} not found. "
                "Using PLM-only path."
            )

    plm_embeddings = plm_embeddings.loc[target_variants]
    if struct_df is not None:
        struct_df = struct_df.loc[target_variants]

        if not struct_df.index.equals(plm_embeddings.index):
            raise RuntimeError(
                "Structural features and PLM embeddings are not aligned "
                "after preprocessing."
            )

    print(
        f"Data aligned. Variants: {len(labels)}, "
        f"PLM Dim: {plm_embeddings.shape[1]}, "
        f"Struct Dim: {0 if struct_df is None else struct_df.shape[1]}"
    )

    return plm_embeddings, struct_df, labels


def process_pt_embeddings(
    embeddings: Dict,
    embeddings_type_pt: str,
) -> Optional[Dict]:
    try:
        if embeddings_type_pt == "average":
            return {
                key: _tensor_to_numpy(value["average"])
                for key, value in embeddings.items()
            }

        if embeddings_type_pt == "mutated":
            return {
                key: _tensor_to_numpy(value["mutated"])
                for key, value in embeddings.items()
            }

        if embeddings_type_pt == "both":
            return {
                key: np.concatenate(
                    (
                        _tensor_to_numpy(value["average"]),
                        _tensor_to_numpy(value["mutated"]),
                    )
                )
                for key, value in embeddings.items()
            }

        print(f"Invalid embeddings_type_pt: {embeddings_type_pt}")
        return None

    except Exception as error:
        print(f"Error while processing .pt embeddings: {error}")
        return None


def load_experimental_embeddings(
    base_path: str,
    embeddings_file_name: str,
    rename_WT: bool = False,
) -> pd.DataFrame:
    file_path = os.path.join(base_path, embeddings_file_name)
    embeddings = pd.read_csv(file_path, index_col=0)
    embeddings.index = _clean_variant_index(embeddings.index)

    if rename_WT:
        embeddings = embeddings.rename(
            index={"WT Wild-type sequence": "WT"}
        )

    return embeddings.sort_index()


def load_experimental_data(
    base_path: str,
    round_file_name: str,
    wt_fasta_path: str,
    single_mutant: bool = True,
) -> pd.DataFrame:
    file_path = os.path.join(base_path, round_file_name)
    dataframe = pd.read_excel(file_path)
    wt_sequence = str(SeqIO.read(wt_fasta_path, "fasta").seq)

    if single_mutant:
        dataframe["updated_variant"] = dataframe["Variant"].apply(
            lambda value: process_variant(value, wt_sequence)
        )
    else:
        dataframe.rename(
            columns={"Variant": "updated_variant"},
            inplace=True,
        )

    return dataframe


def process_variant(variant: str, wt_sequence: str) -> str:
    if variant == "WT" or not any(
        character.isdigit() for character in str(variant)
    ):
        return variant

    try:
        position_string = "".join(filter(str.isdigit, str(variant)))
        position = int(position_string)
        wt_aa = wt_sequence[position - 1]
        return wt_aa + str(variant)
    except Exception:
        return variant


def create_iteration_dataframes(
    df_list: List[pd.DataFrame],
    expected_variants: List[str],
) -> Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None]]:
    processed_dfs = []

    for round_num, dataframe in enumerate(df_list, start=1):
        dataframe_copy = dataframe.copy()

        if round_num == 1:
            dataframe_copy.loc[
                dataframe_copy["updated_variant"] == "WT",
                "iteration",
            ] = 0
        else:
            dataframe_copy = dataframe_copy[
                dataframe_copy["updated_variant"] != "WT"
            ]

        dataframe_copy.loc[
            dataframe_copy["updated_variant"] != "WT",
            "iteration",
        ] = round_num
        dataframe_copy.rename(
            columns={"updated_variant": "variant"},
            inplace=True,
        )
        processed_dfs.append(dataframe_copy)

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    if has_duplicates(combined_df):
        return None, None

    iteration_df = combined_df[["variant", "iteration"]]
    labels = combined_df[["variant", "activity", "iteration"]].copy()
    labels["activity_binary"] = (
        labels["activity"] >= 1
    ).astype(int)

    activity_min = labels["activity"].min()
    activity_max = labels["activity"].max()
    labels["activity_scaled"] = (
        (labels["activity"] - activity_min)
        / (activity_max - activity_min)
        if activity_max > activity_min
        else 0.0
    )

    labels = add_missing_variants(labels, expected_variants)
    labels = (
        labels.set_index("variant")
        .reindex(expected_variants)
        .reset_index()
    )
    return iteration_df, labels


def has_duplicates(dataframe: pd.DataFrame) -> bool:
    duplicates = dataframe[
        dataframe.duplicated(subset=["variant"], keep=False)
    ]
    if not duplicates.empty:
        print("Duplicates found in variant column.")
        return True
    return False


def add_missing_variants(
    dataframe: pd.DataFrame,
    expected_variants: List[str],
) -> pd.DataFrame:
    existing_variants = set(dataframe["variant"])
    missing_variants = [
        variant
        for variant in expected_variants
        if variant not in existing_variants
    ]

    if not missing_variants:
        return dataframe

    missing_df = pd.DataFrame(
        {
            "variant": missing_variants,
            "activity": np.nan,
            "activity_binary": np.nan,
            "activity_scaled": np.nan,
            "iteration": np.nan,
        }
    )
    return pd.concat([dataframe, missing_df], ignore_index=True)
