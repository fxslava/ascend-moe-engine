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
# Fetch the DeepSeek-V4-Flash metadata and the weight shard that actually holds
# the Layer 0 tensors, for apps/dsv4_layer0_emu_bench.
#
# WHICH SHARD HOLDS LAYER 0
# -------------------------
# Not the first one. `model.safetensors.index.json` maps
# `model-00001-of-00046.safetensors` to exactly ONE tensor -- `embed.weight`,
# BF16 [129280, 4096], 1,059,061,760 bytes -- and every `layers.0.*` tensor
# (1565 of them) to `model-00002-of-00046.safetensors`. So shard 1 is the
# embedding table and carries no layer weight at all; shard 2 is the one the
# Layer 0 pipeline reads. This script fetches shard 2 by default and takes
# `--shard N` for any other.
#
# SLICE MODE (the default) vs --full
# ----------------------------------
# Every tensor the Layer 0 emulation actually reads -- the two mHC triples
# (hc_attn_fn/scale/base, hc_ffn_fn/scale/base), attn_norm, ffn_norm, q_norm,
# kv_norm, ffn.gate.weight and the five E8M0 scale blocks -- lives in the first
# ~11.5 MB of shard 2's data segment. The FP8 projection matrices start at byte
# 212,802,904. So the default mode issues a single ranged GET for the header
# plus a leading window (16 MiB), which is ~0.4% of the 3.5 GB shard and
# finishes in seconds.
#
# A truncated safetensors file is still a WELL-FORMED one for a reader that
# honours data_offsets: `safetensors_view.hpp` parses the complete header and
# marks any tensor whose bytes fall past EOF as unavailable rather than
# faulting. `--full` downloads the entire shard when the FP8 projections are
# wanted too.
#
# USAGE
#   scripts/download_dsv4_layer0.sh                 # metadata + 16 MiB of shard 2
#   scripts/download_dsv4_layer0.sh --full          # metadata + all of shard 2
#   scripts/download_dsv4_layer0.sh --full --shard 1   # the embedding shard
#   scripts/download_dsv4_layer0.sh --slice-mib 64  # a wider leading window
#   DSV4_DATA_DIR=/mnt/big/dsv4 scripts/download_dsv4_layer0.sh
# -----------------------------------------------------------------------------

set -euo pipefail

REPO="deepseek-ai/DeepSeek-V4-Flash"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${DSV4_DATA_DIR:-${REPO_ROOT}/data/dsv4_layer0}"

MODE="slice"
SLICE_MIB=16
SHARD=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)       MODE="full"; shift ;;
    --slice)      MODE="slice"; shift ;;
    --slice-mib)  SLICE_MIB="${2:?--slice-mib needs a value}"; shift 2 ;;
    --shard)      SHARD="${2:?--shard needs a value}"; shift 2 ;;
    --dir)        TARGET_DIR="${2:?--dir needs a value}"; shift 2 ;;
    -h|--help)    sed -n '19,56p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

if ! [[ "${SHARD}" =~ ^[0-9]+$ ]] || (( SHARD < 1 || SHARD > 46 )); then
  echo "error: --shard must be an integer in 1..46, got '${SHARD}'" >&2
  exit 2
fi
if ! [[ "${SLICE_MIB}" =~ ^[0-9]+$ ]] || (( SLICE_MIB < 1 )); then
  echo "error: --slice-mib must be a positive integer, got '${SLICE_MIB}'" >&2
  exit 2
fi

SHARD_FILE="$(printf 'model-%05d-of-00046.safetensors' "${SHARD}")"
METADATA=(config.json model.safetensors.index.json)

mkdir -p "${TARGET_DIR}"
echo "repo        : ${REPO}"
echo "target dir  : ${TARGET_DIR}"
echo "shard       : ${SHARD_FILE}"
echo "mode        : ${MODE}$([[ ${MODE} == slice ]] && echo " (${SLICE_MIB} MiB leading window)")"
echo

have() { command -v "$1" >/dev/null 2>&1; }

# `huggingface-cli download` was renamed `hf download` in huggingface_hub 0.34.
hf_download() {
  local -a files=("$@")
  if have hf; then
    hf download "${REPO}" "${files[@]}" --local-dir "${TARGET_DIR}"
  else
    huggingface-cli download "${REPO}" "${files[@]}" \
      --local-dir "${TARGET_DIR}" --local-dir-use-symlinks False
  fi
}

# Content-Length of a repo file, or empty when the probe fails. HF serves the
# LFS blob off a redirect, so -L is required for the size to be the real one.
remote_size() {
  curl --fail --location --silent --head "${BASE_URL}/$1" 2>/dev/null \
    | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }'
}

local_size() { [[ -f "$1" ]] && stat -c %s "$1" || echo 0; }

# Fresh GET through a .part file. Used for the small metadata documents: a
# plain `-C -` against an already-complete file makes the server answer
# 416 Range Not Satisfiable, which curl reports as a hard failure.
curl_fresh() {
  local file="$1"
  local part="${TARGET_DIR}/${file}.part"
  echo "curl: ${file}"
  rm -f "${part}"
  curl --fail --location --retry 5 --retry-delay 2 \
       -o "${part}" "${BASE_URL}/${file}"
  mv -f "${part}" "${TARGET_DIR}/${file}"
}

# Resumable GET for the multi-gigabyte shard. Skips the transfer outright when
# the local file is already the full remote length, so re-running the script is
# cheap and never trips the 416 above.
curl_resume() {
  local file="$1"
  local dest="${TARGET_DIR}/${file}"
  local remote
  remote="$(remote_size "${file}")"
  local have
  have="$(local_size "${dest}")"
  if [[ -n "${remote}" ]] && (( have == remote )); then
    echo "already complete (${have} bytes): ${file}"
    return 0
  fi
  if (( have > 0 )); then
    echo "curl (resuming at ${have} of ${remote:-?} bytes): ${file}"
  else
    echo "curl (${remote:-?} bytes): ${file}"
  fi
  curl --fail --location --retry 5 --retry-delay 2 \
       -C - -o "${dest}" "${BASE_URL}/${file}"
}

# A ranged GET for the leading window. Not resumable (-C - and an explicit
# Range are mutually exclusive), so it writes to a .part file and renames on
# success -- a half-finished window must never be mistaken for a good one.
curl_slice() {
  local file="$1"
  local bytes="$2"
  local part="${TARGET_DIR}/${file}.part"
  local dest="${TARGET_DIR}/${file}"
  local remote
  remote="$(remote_size "${file}")"
  # Never ask for more than the file holds: a Range whose start is past EOF is
  # another 416. A shard smaller than the window just arrives whole.
  if [[ -n "${remote}" ]] && (( bytes > remote )); then
    bytes="${remote}"
  fi
  if (( "$(local_size "${dest}")" >= bytes )); then
    echo "already have >= ${bytes} bytes: ${file}"
    return 0
  fi
  echo "curl (ranged 0-$((bytes - 1))): ${file}"
  rm -f "${part}"
  curl --fail --location --retry 5 --retry-delay 2 \
       -H "Range: bytes=0-$((bytes - 1))" \
       -o "${part}" "${BASE_URL}/${file}"
  mv -f "${part}" "${dest}"
}

# ---- metadata ---------------------------------------------------------------
if have hf || have huggingface-cli; then
  echo "metadata via huggingface-cli"
  hf_download "${METADATA[@]}"
else
  echo "huggingface-cli not found; metadata via curl"
  for f in "${METADATA[@]}"; do curl_fresh "${f}"; done
fi
echo

# ---- the shard --------------------------------------------------------------
# huggingface-cli has no ranged-download interface, so slice mode always uses
# curl; full mode prefers the CLI for its hash check and resume.
if [[ "${MODE}" == "full" ]]; then
  if have hf || have huggingface-cli; then
    hf_download "${SHARD_FILE}"
  else
    curl_resume "${SHARD_FILE}"
  fi
else
  curl_slice "${SHARD_FILE}" "$((SLICE_MIB * 1024 * 1024))"
fi

# ---- report -----------------------------------------------------------------
echo
echo "fetched into ${TARGET_DIR}:"
for f in "${METADATA[@]}" "${SHARD_FILE}"; do
  if [[ -f "${TARGET_DIR}/${f}" ]]; then
    printf '  %-40s %12s bytes\n' "${f}" "$(stat -c %s "${TARGET_DIR}/${f}")"
  else
    printf '  %-40s %12s\n' "${f}" "MISSING"
  fi
done

if [[ "${MODE}" == "slice" ]]; then
  cat <<'NOTE'

NOTE: this is a LEADING SLICE of the shard, not the whole file. The emulation
bench reads only the tensors that fall inside it and reports the rest as
"not in the downloaded slice". Re-run with --full for the FP8 projections.
NOTE
fi
