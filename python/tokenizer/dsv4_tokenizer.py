#
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
#

"""DeepSeek-V4-Flash tokenizer: text -> token ids -> text, with ZERO torch.

Wraps the checkpoint's ``tokenizer.json`` (HuggingFace fast-tokenizer
format) through the Rust-backed ``tokenizers`` package -- no PyTorch, no
torch_npu, no transformers import chain. The checkpoint directory is found
from ``--model-dir``, then ``$DSV4_MODEL_DIR``, then the usual mount points.

CLI::

    python -m tokenizer.dsv4_tokenizer "Hello DeepSeek"
    python -m tokenizer.dsv4_tokenizer --decode "15436,306,394"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_MODEL_DIRS = (
    os.environ.get("DSV4_MODEL_DIR"),
    "/mnt/c/models/DeepSeek-V4-Flash",
    "C:/models/DeepSeek-V4-Flash",
)


def find_tokenizer_json(explicit: str | None = None) -> Path:
    """Locates tokenizer.json: explicit path/dir first, then the defaults."""
    candidates = []
    if explicit:
        candidate = Path(explicit)
        candidates.append(candidate if candidate.is_file() else candidate / "tokenizer.json")
    for directory in DEFAULT_MODEL_DIRS:
        if directory:
            candidates.append(Path(directory) / "tokenizer.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"tokenizer.json not found (searched: {searched}); pip install tokenizers first")


class Dsv4Tokenizer:
    """Thin wrapper: encode text to ids, decode ids back to text."""

    def __init__(self, model_dir_or_file: str | None = None):
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415 -- lazy: keeps --help import-free
        except ImportError as error:  # pragma: no cover -- environment dependent
            raise SystemExit(
                "the 'tokenizers' package is required (it is Rust-backed and torch-free): "
                "pip install tokenizers"
            ) from error
        self.path = find_tokenizer_json(model_dir_or_file)
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encodes text into token ids (no special tokens added)."""
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        """Decodes token ids back into text."""
        return self._tokenizer.decode(ids, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tokenizer.dsv4_tokenizer",
        description="Encode text to DeepSeek-V4-Flash token ids (and back), zero torch.",
    )
    parser.add_argument("text", nargs="?", help="text to encode")
    parser.add_argument("--model-dir", default=None, help="checkpoint directory holding tokenizer.json")
    parser.add_argument("--decode", default=None, help="comma-separated ids to decode instead of encoding")
    arguments = parser.parse_args(argv)

    tokenizer = Dsv4Tokenizer(arguments.model_dir)
    if arguments.decode is not None:
        ids = [int(piece) for piece in arguments.decode.split(",") if piece.strip()]
        text = tokenizer.decode(ids)
        print(f"ids   : {ids}")
        print(f"decode: {text!r}")
        return 0
    if arguments.text is None:
        parser.error("either TEXT or --decode is required")
    ids = tokenizer.encode(arguments.text)
    round_trip = tokenizer.decode(ids)
    print(f"text  : {arguments.text!r}")
    print(f"ids   : {ids}")
    print(f"count : {len(ids)} (vocab {tokenizer.vocab_size})")
    print(f"decode: {round_trip!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
