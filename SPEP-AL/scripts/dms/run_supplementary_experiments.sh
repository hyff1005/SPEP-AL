#!/bin/bash
set -eo pipefail


export PS1="${PS1-}"

set +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] 找不到 conda.sh，请检查 conda 安装路径。"
    exit 1
fi

conda activate plm
set -u

# ------------------------------------------------------------
# 基础路径
# ------------------------------------------------------------
BASE_DIR=${BASE_DIR:-/code/EvolvePro}
SCRIPT=${SCRIPT:-${BASE_DIR}/scripts/struct/dms_main.py}

EMB_DIR=${EMB_DIR:-${BASE_DIR}/output/plm/esm}
LBL_DIR=${LBL_DIR:-${BASE_DIR}/output/dms}
STRUCT_DIR=${STRUCT_DIR:-${BASE_DIR}/output/struct}

OUT_ROOT=${OUT_ROOT:-${BASE_DIR}/output/formal_experiments}
LOG_ROOT=${LOG_ROOT:-${OUT_ROOT}/_logs}
ABLATION_ROOT=${ABLATION_ROOT:-${BASE_DIR}/output/struct_ablation}

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH:-}

MODEL_NAME="esm2_t36_3B_UR50D"
EMB_FILE_TYPE="csv"
MEASURED_VAR="activity"
FIRST_ROUND_STRATEGY="random"
REGRESSION_TYPE="randomforest"

NUM_SIM=${NUM_SIM:-10}
NUM_ITER=${NUM_ITER:-10}
DEFAULT_NUM_MUT=${DEFAULT_NUM_MUT:-16}
DEFAULT_NUM_FINAL=${DEFAULT_NUM_FINAL:-16}

FORCE_RERUN=${FORCE_RERUN:-0}
RUN_GROUPS=${RUN_GROUPS:-structure,pareto,sensitivity}
RUN_EXTENDED=${RUN_EXTENDED:-1}

DATASETS=(
    "brenan"
    "stiffler"
    "doud"
    "haddox"
    "giacomelli"
    "jones"
    "kelsic"
    "lee"
    "markin"
    "zikv_E"
    "cas12f"
    "cov2_S"
)

group_enabled() {
    local group="$1"
    [[ ",${RUN_GROUPS}," == *",${group},"* ]]
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

command -v python3 >/dev/null 2>&1 || die "python3 不存在。"
[ -f "${SCRIPT}" ] || die "dms_main.py 不存在：${SCRIPT}"
[ -d "${EMB_DIR}" ] || die "嵌入目录不存在：${EMB_DIR}"
[ -d "${LBL_DIR}" ] || die "标签目录不存在：${LBL_DIR}"
[ -d "${STRUCT_DIR}" ] || die "结构目录不存在：${STRUCT_DIR}"

PRECHECK_FAILED=0
for dataset_name in "${DATASETS[@]}"; do
    emb_file="${EMB_DIR}/${dataset_name}_${MODEL_NAME}.${EMB_FILE_TYPE}"
    label_file="${LBL_DIR}/${dataset_name}_labels.csv"
    struct_file="${STRUCT_DIR}/${dataset_name}_struct.csv"

    [ -f "${emb_file}" ] || {
        echo "[ERROR] 缺少嵌入：${emb_file}"
        PRECHECK_FAILED=1
    }
    [ -f "${label_file}" ] || {
        echo "[ERROR] 缺少标签：${label_file}"
        PRECHECK_FAILED=1
    }
    [ -f "${struct_file}" ] || {
        echo "[ERROR] 缺少结构特征：${struct_file}"
        PRECHECK_FAILED=1
    }
done

[ "${PRECHECK_FAILED}" -eq 0 ] || die "输入文件预检查失败。"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${ABLATION_ROOT}"

if group_enabled "structure"; then
    echo "============================================================"
    echo "[PREP] 构造结构消融数据"
    echo "============================================================"

    export STRUCT_DIR ABLATION_ROOT
    export DATASETS_JOINED="${DATASETS[*]}"

    python3 - <<'PY'
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

struct_dir = Path(os.environ["STRUCT_DIR"])
root = Path(os.environ["ABLATION_ROOT"])
datasets = os.environ["DATASETS_JOINED"].split()

wt_cols = [
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
]

delta_cols = [
    "delta_site_relative_sasa",
    "delta_local_mean_relative_sasa",
    "delta_contact_count_8A",
    "mutation_site_ca_displacement",
    "local_ca_rmsd",
    "local_distance_rmsd",
    "local_max_ca_displacement",
    "delta_local_radius_of_gyration",
]

all_cols = wt_cols + delta_cols

aa_props = {
    "A": (88.6, 1.8, 0.0), "R": (173.4, -4.5, 1.0),
    "N": (114.1, -3.5, 0.0), "D": (111.1, -3.5, -1.0),
    "C": (108.5, 2.5, 0.0), "Q": (143.8, -3.5, 0.0),
    "E": (138.4, -3.5, -1.0), "G": (60.1, -0.4, 0.0),
    "H": (153.2, -3.2, 0.1), "I": (166.7, 4.5, 0.0),
    "L": (166.7, 3.8, 0.0), "K": (168.6, -3.9, 1.0),
    "M": (162.9, 1.9, 0.0), "F": (189.9, 2.8, 0.0),
    "P": (112.7, -1.6, 0.0), "S": (89.0, -0.8, 0.0),
    "T": (116.1, -0.7, 0.0), "W": (227.8, -0.9, 0.0),
    "Y": (193.6, -1.3, 0.0), "V": (140.0, 4.2, 0.0),
}

dirs = {
    "wt11": root / "wt11",
    "delta8": root / "delta8",
    "position_physchem4": root / "position_physchem4",
    "shuffled19": root / "shuffled19",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

summary_rows = []

for dataset in datasets:
    src_struct = struct_dir / f"{dataset}_struct.csv"
    src_qc = struct_dir / f"{dataset}_struct_qc.csv"

    struct = pd.read_csv(src_struct, index_col=0)
    struct.index = struct.index.map(lambda x: str(x).strip())
    struct = struct[~struct.index.duplicated(keep="first")]

    missing = [c for c in all_cols if c not in struct.columns]
    if missing:
        raise RuntimeError(
            f"{dataset}: 原结构文件缺少预期Struct19列：{missing}"
        )

    # WT11
    struct[wt_cols].to_csv(dirs["wt11"] / f"{dataset}_struct.csv")

    # Delta8
    struct[delta_cols].to_csv(dirs["delta8"] / f"{dataset}_struct.csv")

    # 位置 + 氨基酸理化性质变化控制
    control = pd.DataFrame(index=struct.index)
    positions = []
    delta_volume = []
    delta_hydro = []
    delta_charge = []

    for variant in struct.index:
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", str(variant).strip().upper())
        if m is None:
            positions.append(np.nan)
            delta_volume.append(0.0)
            delta_hydro.append(0.0)
            delta_charge.append(0.0)
            continue

        wt, pos, mt = m.group(1), int(m.group(2)), m.group(3)
        wt_p = aa_props.get(wt, (0.0, 0.0, 0.0))
        mt_p = aa_props.get(mt, (0.0, 0.0, 0.0))

        positions.append(float(pos))
        delta_volume.append(float(mt_p[0] - wt_p[0]))
        delta_hydro.append(float(mt_p[1] - wt_p[1]))
        delta_charge.append(float(mt_p[2] - wt_p[2]))

    control["mutation_position"] = positions
    control["delta_aa_volume"] = delta_volume
    control["delta_hydrophobicity"] = delta_hydro
    control["delta_charge"] = delta_charge
    control.to_csv(
        dirs["position_physchem4"] / f"{dataset}_struct.csv"
    )

    # Shuffled19：WT保留原位；其余variant确定性打乱。
    shuffled = struct[all_cols].copy()
    mutant_indices = [
        idx for idx in shuffled.index
        if str(idx).strip().upper() not in {"WT", "WILDTYPE", "WILD_TYPE"}
    ]

    seed = int(
        hashlib.sha256(f"evolvepro_struct_shuffle::{dataset}".encode()).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    source_indices = list(rng.permutation(mutant_indices))

    if mutant_indices:
        shuffled.loc[mutant_indices, all_cols] = (
            struct.loc[source_indices, all_cols].to_numpy()
        )

    shuffled.to_csv(dirs["shuffled19"] / f"{dataset}_struct.csv")

    # WT11/Delta8沿用原QC。
    if src_qc.exists():
        qc = pd.read_csv(src_qc, index_col=0)
        qc.index = qc.index.map(lambda x: str(x).strip())
        qc = qc[~qc.index.duplicated(keep="first")]

        qc.to_csv(dirs["wt11"] / f"{dataset}_struct_qc.csv")
        qc.to_csv(dirs["delta8"] / f"{dataset}_struct_qc.csv")

        # Shuffled19 的QC跟随被打乱的结构行一起移动。
        qc_aligned = qc.reindex(struct.index)
        shuffled_qc = qc_aligned.copy()
        if mutant_indices:
            shuffled_qc.loc[mutant_indices, :] = (
                qc_aligned.loc[source_indices, :].to_numpy()
            )
        shuffled_qc.to_csv(
            dirs["shuffled19"] / f"{dataset}_struct_qc.csv"
        )

    summary_rows.append(
        {
            "dataset": dataset,
            "n_struct_rows": len(struct),
            "wt11_dim": len(wt_cols),
            "delta8_dim": len(delta_cols),
            "position_physchem_dim": control.shape[1],
            "shuffle_seed": seed,
            "qc_available": src_qc.exists(),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(root / "struct_ablation_prepare_summary.csv", index=False)

print(summary.to_string(index=False))
print(f"[DONE] 结构消融数据已写入：{root}")
PY
fi

EXP_NAMES=()
EXP_GROUPS=()
EXP_STRUCT_PATHS=()
EXP_EMBEDDINGS=()
EXP_STRATEGIES=()
EXP_NUM_MUTS=()
EXP_NUM_FINALS=()
EXP_POOLS=()
EXP_LAMBDAS=()
EXP_RULES=()
EXP_QRETS=()
EXP_LOCALS=()
EXP_LOCAL_STEPS=()
EXP_ENTROPY_BASES=()
EXP_TOP1S=()
EXP_DESCRIPTIONS=()

add_exp() {
    EXP_NAMES+=("$1")
    EXP_GROUPS+=("$2")
    EXP_STRUCT_PATHS+=("$3")
    EXP_EMBEDDINGS+=("$4")
    EXP_STRATEGIES+=("$5")
    EXP_NUM_MUTS+=("$6")
    EXP_NUM_FINALS+=("$7")
    EXP_POOLS+=("$8")
    EXP_LAMBDAS+=("$9")
    EXP_RULES+=("${10}")
    EXP_QRETS+=("${11}")
    EXP_LOCALS+=("${12}")
    EXP_LOCAL_STEPS+=("${13}")
    EXP_ENTROPY_BASES+=("${14}")
    EXP_TOP1S+=("${15}")
    EXP_DESCRIPTIONS+=("${16}")
}

DEFAULT_LAMBDAS="0.0 0.05 0.10 0.20 0.40 0.80 1.20 1.60 2.00"

if group_enabled "structure"; then
    add_exp \
        "suppA_pca10_wt11_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "structure" \
        "${ABLATION_ROOT}/wt11" \
        "embeddings_pca_10_struct_fusion" \
        "topn" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "PCA10 + WT环境11维 + RandomForest + TopN"

    add_exp \
        "suppA_pca10_delta8_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "structure" \
        "${ABLATION_ROOT}/delta8" \
        "embeddings_pca_10_struct_fusion" \
        "topn" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "PCA10 + 结构差分8维 + RandomForest + TopN"

    add_exp \
        "suppA_pca10_posphys4_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "structure" \
        "${ABLATION_ROOT}/position_physchem4" \
        "embeddings_pca_10_struct_fusion" \
        "topn" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "PCA10 + 突变位置/理化性质4维控制 + RandomForest + TopN"

    add_exp \
        "suppA_pca10_shuffled19_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "structure" \
        "${ABLATION_ROOT}/shuffled19" \
        "embeddings_pca_10_struct_fusion" \
        "topn" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "PCA10 + 随机打乱Struct19 + RandomForest + TopN"

    add_exp \
        "suppA_struct19_only_rf_topn_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "structure" \
        "${STRUCT_DIR}" \
        "embeddings_struct_only" \
        "topn" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "仅Struct19 + RandomForest + TopN"
fi

if group_enabled "pareto"; then
    add_exp \
        "suppB_struct19_fixedlambda040_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "pareto" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "0.40" "max_entropy_under_quality" "0.0" "true" "50" "batch" "true" \
        "固定lambda=0.40，不构建多lambda折中集合"

    add_exp \
        "suppB_struct19_maxentropy_q95_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "pareto" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "max_entropy_under_quality" "0.95" "true" "50" "batch" "true" \
        "质量保留率>=0.95条件下最大化位置熵"

    add_exp \
        "suppB_struct19_pareto_no_local_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "pareto" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "false" "0" "batch" "true" \
        "关闭一换一局部搜索"

    add_exp \
        "suppB_struct19_pareto_no_top1_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "pareto" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "false" \
        "不强制保留预测Top1"

    if [ "${RUN_EXTENDED}" -eq 1 ]; then
        add_exp \
            "suppB_struct19_pareto_knee_q95_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
            "pareto" \
            "${STRUCT_DIR}" \
            "embeddings_pca_10_struct_fusion" \
            "position_entropy_pareto" \
            "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
            "4.0" "${DEFAULT_LAMBDAS}" "knee" "0.95" "true" "50" "batch" "true" \
            "使用Pareto膝点规则选择最终批次"

        add_exp \
            "suppB_struct19_pareto_no_quality_guard_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
            "pareto" \
            "${STRUCT_DIR}" \
            "embeddings_pca_10_struct_fusion" \
            "position_entropy_pareto" \
            "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
            "4.0" "${DEFAULT_LAMBDAS}" "product" "0.0" "true" "50" "batch" "true" \
            "关闭最低预测质量保留约束"
    fi
fi

if group_enabled "sensitivity"; then
    add_exp \
        "suppC_struct19_pareto_pool2_q95_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "sensitivity" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "2.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "候选池倍数2，N=16时Top32"

    add_exp \
        "suppC_struct19_pareto_pool8_q95_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "sensitivity" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "8.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "候选池倍数8，N=16时Top128"

    add_exp \
        "suppC_struct19_pareto_pool4_q90_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "sensitivity" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.90" "true" "50" "batch" "true" \
        "最低预测质量保留率0.90"

    add_exp \
        "suppC_struct19_pareto_pool4_q98_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "sensitivity" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "position_entropy_pareto" \
        "${DEFAULT_NUM_MUT}" "${DEFAULT_NUM_FINAL}" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.98" "true" "50" "batch" "true" \
        "最低预测质量保留率0.98"

    add_exp \
        "suppC_struct19_batch8_topn_vs_pareto_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "sensitivity" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "topn position_entropy_pareto" \
        "8" "8" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "批次大小8：同文件比较TopN与Pareto"

    add_exp \
        "suppC_struct19_batch32_topn_vs_pareto_activity_sim${NUM_SIM}_iter${NUM_ITER}" \
        "sensitivity" \
        "${STRUCT_DIR}" \
        "embeddings_pca_10_struct_fusion" \
        "topn position_entropy_pareto" \
        "32" "32" \
        "4.0" "${DEFAULT_LAMBDAS}" "product" "0.95" "true" "50" "batch" "true" \
        "批次大小32：同文件比较TopN与Pareto"
fi

[ "${#EXP_NAMES[@]}" -gt 0 ] || die "没有启用任何实验组：RUN_GROUPS=${RUN_GROUPS}"

MANIFEST=${OUT_ROOT}/supplement_experiments_manifest.tsv
{
    printf "experiment\tgroup\tstruct_path\tembedding_types\tlearning_strategies\tnum_mutants\tnum_final\tpool_multiplier\tlambda_values\tbalance_rule\tquality_retention\tlocal_search\tlocal_steps\tentropy_base\tforce_top1\tdescription\n"
    for i in "${!EXP_NAMES[@]}"; do
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${EXP_NAMES[$i]}" \
            "${EXP_GROUPS[$i]}" \
            "${EXP_STRUCT_PATHS[$i]}" \
            "${EXP_EMBEDDINGS[$i]}" \
            "${EXP_STRATEGIES[$i]}" \
            "${EXP_NUM_MUTS[$i]}" \
            "${EXP_NUM_FINALS[$i]}" \
            "${EXP_POOLS[$i]}" \
            "${EXP_LAMBDAS[$i]}" \
            "${EXP_RULES[$i]}" \
            "${EXP_QRETS[$i]}" \
            "${EXP_LOCALS[$i]}" \
            "${EXP_LOCAL_STEPS[$i]}" \
            "${EXP_ENTROPY_BASES[$i]}" \
            "${EXP_TOP1S[$i]}" \
            "${EXP_DESCRIPTIONS[$i]}"
    done
} > "${MANIFEST}"

echo "============================================================"
echo "[INFO] 补充实验数量        : ${#EXP_NAMES[@]}"
echo "[INFO] 数据集数量          : ${#DATASETS[@]}"
echo "[INFO] 每组模拟次数        : ${NUM_SIM}"
echo "[INFO] 主动学习轮数        : ${NUM_ITER}"
echo "[INFO] 实验组              : ${RUN_GROUPS}"
echo "[INFO] 扩展组件实验        : ${RUN_EXTENDED}"
echo "[INFO] FORCE_RERUN         : ${FORCE_RERUN}"
echo "[INFO] 实验清单            : ${MANIFEST}"
echo "============================================================"

FAILED_ITEMS=()
TOTAL_RUN=0
TOTAL_SKIP=0

run_one() {
    local exp_name="$1"
    local struct_path="$2"
    local embedding_string="$3"
    local strategy_string="$4"
    local num_mut="$5"
    local num_final="$6"
    local pool="$7"
    local lambda_string="$8"
    local rule="$9"
    local qret="${10}"
    local local_search="${11}"
    local local_steps="${12}"
    local entropy_base="${13}"
    local force_top1="${14}"
    local dataset_name="${15}"

    local result_dir="${OUT_ROOT}/${exp_name}"
    local log_dir="${LOG_ROOT}/${exp_name}"
    local result_file="${result_dir}/${dataset_name}_${MODEL_NAME}_${exp_name}.csv"
    local log_file="${log_dir}/${dataset_name}.out"
    local time_file="${log_dir}/${dataset_name}.time.txt"

    mkdir -p "${result_dir}" "${log_dir}"

    if [ "${FORCE_RERUN}" -ne 1 ] && [ -s "${result_file}" ]; then
        echo "[SKIP] ${exp_name} / ${dataset_name}"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi

    read -r -a embedding_arr <<< "${embedding_string}"
    read -r -a strategy_arr <<< "${strategy_string}"
    read -r -a lambda_arr <<< "${lambda_string}"

    cmd=(
        python3 -u "${SCRIPT}"

        --dataset_name "${dataset_name}"
        --experiment_name "${exp_name}"
        --model_name "${MODEL_NAME}"

        --embeddings_path "${EMB_DIR}"
        --labels_path "${LBL_DIR}"
        --struct_path "${struct_path}"

        --num_simulations "${NUM_SIM}"
        --num_iterations "${NUM_ITER}"
        --measured_var "${MEASURED_VAR}"
        --learning_strategies "${strategy_arr[@]}"
        --num_mutants_per_round "${num_mut}"
        --num_final_round_mutants "${num_final}"
        --first_round_strategies "${FIRST_ROUND_STRATEGY}"

        --embedding_types "${embedding_arr[@]}"
        --pca_components 10
        --regression_types "${REGRESSION_TYPE}"

        --embeddings_file_type "${EMB_FILE_TYPE}"
        --output_dir "${result_dir}"

        --ensemble_size 1
        --bootstrap_sample_frac 1.0

        --ucb_beta 0.0
        --novelty_weight 0.0
        --redundancy_weight 0.0
        --thompson_scale 1.0
        --mixed_exploit_frac 1.0

        --enable_tail_weighting false
        --enable_head_rerank false
        --enable_success_head false
        --success_weight 0.0
        --success_filter_quantile 0.0
        --success_mode off

        --pareto_pool_multiplier "${pool}"
        --pareto_lambda_values "${lambda_arr[@]}"
        --pareto_balance_rule "${rule}"
        --pareto_min_quality_retention "${qret}"
        --pareto_local_search "${local_search}"
        --pareto_local_search_steps "${local_steps}"
        --pareto_entropy_base "${entropy_base}"
        --pareto_force_top1 "${force_top1}"
    )

    echo "------------------------------------------------------------"
    echo "[RUN] ${exp_name}"
    echo "      dataset       = ${dataset_name}"
    echo "      embedding     = ${embedding_string}"
    echo "      strategy      = ${strategy_string}"
    echo "      batch         = ${num_mut}"
    echo "      pool/qret     = ${pool}/${qret}"
    echo "      rule          = ${rule}"
    echo "      local/top1    = ${local_search}/${force_top1}"
    echo "      log           = ${log_file}"
    echo "------------------------------------------------------------"

    set +e
    if [ -x /usr/bin/time ]; then
        /usr/bin/time \
            -f "elapsed_sec=%e\nuser_sec=%U\nsystem_sec=%S\nmax_rss_kb=%M" \
            -o "${time_file}" \
            "${cmd[@]}" > "${log_file}" 2>&1
        exit_code=$?
    else
        start_epoch=$(date +%s)
        "${cmd[@]}" > "${log_file}" 2>&1
        exit_code=$?
        end_epoch=$(date +%s)
        echo "elapsed_sec=$((end_epoch - start_epoch))" > "${time_file}"
    fi
    set -e

    TOTAL_RUN=$((TOTAL_RUN + 1))

    if [ "${exit_code}" -ne 0 ]; then
        echo "[FAIL] ${exp_name} / ${dataset_name}, exit=${exit_code}"
        echo "       日志：${log_file}"
        FAILED_ITEMS+=("${exp_name}/${dataset_name}")
        return 0
    fi

    if [ ! -s "${result_file}" ]; then
        echo "[FAIL] 命令返回成功，但结果文件不存在或为空：${result_file}"
        FAILED_ITEMS+=("${exp_name}/${dataset_name}")
        return 0
    fi

    local line_count
    line_count=$(wc -l < "${result_file}")
    if [ "${line_count}" -lt 2 ]; then
        echo "[FAIL] 结果文件行数异常：${result_file}, lines=${line_count}"
        FAILED_ITEMS+=("${exp_name}/${dataset_name}")
        return 0
    fi

    echo "[DONE] ${exp_name} / ${dataset_name}, lines=${line_count}"
}

for i in "${!EXP_NAMES[@]}"; do
    exp_name="${EXP_NAMES[$i]}"

    echo
    echo "############################################################"
    echo "[EXPERIMENT $((i + 1))/${#EXP_NAMES[@]}] ${exp_name}"
    echo "[DESC] ${EXP_DESCRIPTIONS[$i]}"
    echo "############################################################"

    for dataset_name in "${DATASETS[@]}"; do
        run_one \
            "${exp_name}" \
            "${EXP_STRUCT_PATHS[$i]}" \
            "${EXP_EMBEDDINGS[$i]}" \
            "${EXP_STRATEGIES[$i]}" \
            "${EXP_NUM_MUTS[$i]}" \
            "${EXP_NUM_FINALS[$i]}" \
            "${EXP_POOLS[$i]}" \
            "${EXP_LAMBDAS[$i]}" \
            "${EXP_RULES[$i]}" \
            "${EXP_QRETS[$i]}" \
            "${EXP_LOCALS[$i]}" \
            "${EXP_LOCAL_STEPS[$i]}" \
            "${EXP_ENTROPY_BASES[$i]}" \
            "${EXP_TOP1S[$i]}" \
            "${dataset_name}"
    done
done

echo
echo "============================================================"
echo "[SUMMARY] 新运行任务数：${TOTAL_RUN}"
echo "[SUMMARY] 跳过已有结果：${TOTAL_SKIP}"
echo "[SUMMARY] Manifest：${MANIFEST}"
echo "[SUMMARY] Results：${OUT_ROOT}"
echo "[SUMMARY] Logs：${LOG_ROOT}"

if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
    FAIL_FILE="${OUT_ROOT}/supplement_failed_items.txt"
    printf "%s\n" "${FAILED_ITEMS[@]}" > "${FAIL_FILE}"
    echo "[ERROR] 失败任务数：${#FAILED_ITEMS[@]}"
    echo "[ERROR] 失败清单：${FAIL_FILE}"
    printf "  - %s\n" "${FAILED_ITEMS[@]}"
    exit 1
fi

echo "[DONE] 所有启用的补充实验均已完成。"
echo "============================================================"
