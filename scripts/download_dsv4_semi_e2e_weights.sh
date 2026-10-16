#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
#
# Fetch exactly the DeepSeek-V4-Flash tensors apps/dsv4_semi_e2e_bench needs,
# by HTTP range, and repack them into ONE compact safetensors file.
#
# WHY RANGES AND NOT A LEADING SLICE
#   download_dsv4_layer0.sh takes a leading window of a shard, which is enough
#   for the Layer 0 mHC weights. It is useless for the MoE experts: in shard 3,
#   expert 0 starts at byte 344,923,480 and expert 7 at 3,150,912,856, so the
#   ~100 MB we want is spread over 3.4 GB. Each tensor is therefore fetched by
#   its own Range request and written into a NEW, valid safetensors file with
#   regenerated offsets -- not a sparse shard-shaped file, whose holes would
#   read as zeros and be indistinguishable from real weights.
#
# WHICH EXPERTS
#   Layers 0-2 are the hash layers ("num_hash_layers": 3): the routed experts
#   are a pure function of the token id, indices = tid2eid[token_id]. So the
#   "first N experts" are not the ones a token activates. This fetches
#   layers.<moe>.ffn.gate.tid2eid, reads the row for --token-id, and downloads
#   exactly those six experts. The id and its expert row are recorded in the
#   output's __metadata__, and the bench asserts the router selects experts
#   that are actually present.
#
# FOOTPRINT
#   ~365 MB on disk. The bench mmaps it and dequantizes per output row, so
#   resident memory stays a few MB above the mapped pages -- see the RSS
#   report the bench prints.
#
# USAGE
#   scripts/download_dsv4_semi_e2e_weights.sh
#   scripts/download_dsv4_semi_e2e_weights.sh --token-id 1234 --moe-layer 2
#   scripts/download_dsv4_semi_e2e_weights.sh --dry-run
#   DSV4_DATA_DIR=/mnt/big/dsv4 scripts/download_dsv4_semi_e2e_weights.sh
# -----------------------------------------------------------------------------

set -euo pipefail

REPO="deepseek-ai/DeepSeek-V4-Flash"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${DSV4_DATA_DIR:-${REPO_ROOT}/data/dsv4_layer0}"
FETCHER="${REPO_ROOT}/scripts/dsv4_fetch_tensors.py"

ATTN_LAYERS="0,2"
MOE_LAYER=1
TOKEN_ID=3
MAX_EXPERTS=8
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attn-layers)  ATTN_LAYERS="${2:?}"; shift 2 ;;
    --moe-layer)    MOE_LAYER="${2:?}"; shift 2 ;;
    --token-id)     TOKEN_ID="${2:?}"; shift 2 ;;
    --max-experts)  MAX_EXPERTS="${2:?}"; shift 2 ;;
    --dir)          TARGET_DIR="${2:?}"; shift 2 ;;
    --dry-run)      DRY_RUN="--dry-run"; shift ;;
    -h|--help)      sed -n '19,48p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

for tool in curl python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} is required" >&2; exit 2; }
done
[[ -f "${FETCHER}" ]] || { echo "error: missing ${FETCHER}" >&2; exit 2; }

mkdir -p "${TARGET_DIR}"
echo "repo        : ${REPO}"
echo "target dir  : ${TARGET_DIR}"
echo "attn layers : ${ATTN_LAYERS}"
echo "moe layer   : ${MOE_LAYER} (token id ${TOKEN_ID})"
echo

# ---- metadata ---------------------------------------------------------------
# Fresh GET through a .part file: a plain `-C -` against an already-complete
# file makes the server answer 416, which curl reports as a hard failure.
for file in config.json model.safetensors.index.json; do
  if [[ -s "${TARGET_DIR}/${file}" ]]; then
    echo "have ${file}"
  else
    echo "fetching ${file}"
    curl --fail --location --silent --show-error --retry 5 --retry-delay 2 \
         -o "${TARGET_DIR}/${file}.part" "${BASE_URL}/${file}"
    mv -f "${TARGET_DIR}/${file}.part" "${TARGET_DIR}/${file}"
  fi
done
echo

# ---- the tensors ------------------------------------------------------------
# -I so the interpreter ignores the environment and the script's own directory
# for imports; the fetcher needs nothing outside the standard library.
OUT="${TARGET_DIR}/dsv4_semi_e2e.safetensors"
echo "assembling ${OUT}"
python3 -I "${FETCHER}" \
  --out "${OUT}" \
  --index "${TARGET_DIR}/model.safetensors.index.json" \
  --config "${TARGET_DIR}/config.json" \
  --attn-layers "${ATTN_LAYERS}" \
  --moe-layer "${MOE_LAYER}" \
  --token-id "${TOKEN_ID}" \
  --max-experts "${MAX_EXPERTS}" \
  ${DRY_RUN}

if [[ -z "${DRY_RUN}" ]]; then
  echo
  printf 'fetched: %-44s %12s bytes\n' \
    "dsv4_semi_e2e.safetensors" "$(stat -c %s "${OUT}")"
  echo
  echo "run:  ./build_emu/apps/dsv4_semi_e2e_bench"
fi
