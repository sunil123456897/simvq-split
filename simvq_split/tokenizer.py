"""
Shared lossless Byte-Level BPE Tokenizer for SimVQ-Split and all baselines.
Zero domain heuristics: starts from 256 raw UTF-8 bytes + 3 control tokens
(<pad>=0, <bos>=1, <eos>=2) and learns byte-pair merges purely by frequency on
training text. Encodes arbitrary UTF-8 strings with zero <unk>.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Dict, Iterable, List, Tuple

# GPT-2 style pre-tokenization split so merges don't cross whitespace/punctuation wildly,
# while still operating purely on UTF-8 bytes of each chunk.
_PRETOKEN_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+"
)


class ByteBPETokenizer:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    BYTE_OFFSET = 3  # IDs 3..258 represent raw bytes 0..255

    def __init__(self, merges: List[Tuple[int, int]] | None = None):
        self.merges: List[Tuple[int, int]] = list(merges) if merges else []
        self._rebuild_tables()

    def _rebuild_tables(self) -> None:
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.id_to_bytes: Dict[int, bytes] = {
            self.PAD_ID: b"",
            self.BOS_ID: b"",
            self.EOS_ID: b"",
        }
        for b in range(256):
            self.id_to_bytes[self.BYTE_OFFSET + b] = bytes([b])
        next_id = self.BYTE_OFFSET + 256
        for a, b in self.merges:
            self.merge_ranks[(a, b)] = next_id
            self.id_to_bytes[next_id] = self.id_to_bytes[a] + self.id_to_bytes[b]
            next_id += 1
        self.vocab_size: int = next_id
        self._chunk_cache: Dict[str, List[int]] = {}

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 2048,
        max_chars: int = 1_500_000,
    ) -> "ByteBPETokenizer":
        assert vocab_size >= cls.BYTE_OFFSET + 256
        num_merges = vocab_size - (cls.BYTE_OFFSET + 256)

        word_counts: Counter[Tuple[int, ...]] = Counter()
        total_chars = 0
        for text in texts:
            if not text:
                continue
            for piece in _PRETOKEN_RE.findall(text):
                b_seq = tuple(cls.BYTE_OFFSET + b for b in piece.encode("utf-8"))
                if b_seq:
                    word_counts[b_seq] += 1
            total_chars += len(text)
            if total_chars >= max_chars:
                break

        words: List[List[int]] = [list(w) for w in word_counts.keys()]
        freqs: List[int] = list(word_counts.values())

        pair_counts: Counter[Tuple[int, int]] = Counter()
        for w, f in zip(words, freqs):
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i + 1])] += f

        merges: List[Tuple[int, int]] = []
        next_id = cls.BYTE_OFFSET + 256

        for _ in range(num_merges):
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 2:
                break
            merges.append(best_pair)
            p0, p1 = best_pair
            del pair_counts[best_pair]

            for idx, w in enumerate(words):
                n = len(w)
                if n < 2:
                    continue
                # Fast check if p0 is in word
                if p0 not in w:
                    continue
                f = freqs[idx]
                # Subtract old pairs for this word
                for i in range(n - 1):
                    pair = (w[i], w[i + 1])
                    if pair != best_pair:
                        pair_counts[pair] -= f
                        if pair_counts[pair] <= 0:
                            pair_counts.pop(pair, None)
                # Merge occurrences of best_pair
                new_w: List[int] = []
                i = 0
                while i < n:
                    if i < n - 1 and w[i] == p0 and w[i + 1] == p1:
                        new_w.append(next_id)
                        i += 2
                    else:
                        new_w.append(w[i])
                        i += 1
                words[idx] = new_w
                # Add new pairs for this word
                for i in range(len(new_w) - 1):
                    pair_counts[(new_w[i], new_w[i + 1])] += f

            next_id += 1

        return cls(merges=merges)

    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached
        ids = [self.BYTE_OFFSET + b for b in chunk.encode("utf-8")]
        while len(ids) >= 2:
            best_idx = -1
            best_new_id = 1_000_000_000
            for i in range(len(ids) - 1):
                nid = self.merge_ranks.get((ids[i], ids[i + 1]))
                if nid is not None and nid < best_new_id:
                    best_new_id = nid
                    best_idx = i
            if best_idx < 0:
                break
            ids = ids[:best_idx] + [best_new_id] + ids[best_idx + 2 :]
        if len(self._chunk_cache) < 65536:
            self._chunk_cache[chunk] = ids
        return ids

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        out: List[int] = []
        if add_bos:
            out.append(self.BOS_ID)
        for chunk in _PRETOKEN_RE.findall(text):
            out.extend(self._encode_chunk(chunk))
        if add_eos:
            out.append(self.EOS_ID)
        return out

    def decode(self, token_ids: Iterable[int]) -> str:
        buf = bytearray()
        for tid in token_ids:
            t = int(tid)
            if t in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            buf.extend(self.id_to_bytes[t])
        return buf.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"merges": [[a, b] for a, b in self.merges]}

    @classmethod
    def from_dict(cls, d: dict) -> "ByteBPETokenizer":
        return cls(merges=[(int(a), int(b)) for a, b in d["merges"]])

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
