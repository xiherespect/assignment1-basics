from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path

import regex

from .pretokenization_example import find_chunk_boundaries

class _RevPair:
    """Reverses pair comparison so the heap tiebreaks by largest pair, matching max() behavior."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: "_RevPair") -> bool:
        return self.pair > other.pair

    def __le__(self, other: "_RevPair") -> bool:
        return self.pair >= other.pair

    def __eq__(self, other: "_RevPair") -> bool:
        return self.pair == other.pair

    def __ge__(self, other: "_RevPair") -> bool:
        return self.pair <= other.pair

    def __gt__(self, other: "_RevPair") -> bool:
        return self.pair < other.pair


PRETOKENIZER_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
BYTE_TOKENS = tuple(bytes([i]) for i in range(256))


def pretokenizer(s: str) -> list[str]:
    return regex.findall(PRETOKENIZER_PATTERN, s)


# 对 special token 去重并排序
def _sorted_special_tokens(special_tokens: list[str] | None) -> list[str]:
    if not special_tokens:
        return []
    return sorted(set(special_tokens), key=lambda token: (-len(token), token))


# 把special token编译成 regex，用来匹配 ？
def _compile_special_token_pattern(
    special_tokens: list[str] | None,
) -> regex.Pattern[str] | None:
    ordered = _sorted_special_tokens(special_tokens)
    if not ordered:
        return None
    return regex.compile("|".join(regex.escape(token) for token in ordered))


# 把special token 当作硬边界切开
def _split_on_special_tokens(text: str, special_tokens: list[str] | None) -> list[str]:
    pattern = _compile_special_token_pattern(special_tokens)
    if pattern is None:
        return [text]
    return [segment for segment in pattern.split(text) if segment]


# 把词转成 byte symbols 比如 b“cat" --> b"c",b"a",b"t"
def _word_to_symbols(word_bytes: bytes) -> tuple[bytes, ...]:
    return tuple(BYTE_TOKENS[b] for b in word_bytes)


# 统计 pre-token 里的相邻pair,
def _pair_occurrences(symbols: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    counts: dict[tuple[bytes, bytes], int] = {}
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# merge
def _merge_pair_in_symbols(
    symbols: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    if len(symbols) < 2:
        return symbols

    left, right = pair
    merged = left + right
    result: list[bytes] = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
            result.append(merged)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


# 统计训练语料里，每个pre-token 出现了多少次
def countword(
    text: str, special_tokens: list[str] | None = None
) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in _split_on_special_tokens(text, special_tokens):
        for token in pretokenizer(segment):
            counts[_word_to_symbols(token.encode("utf-8"))] += 1
    return counts


# 子进程worker：读取文件的某一块并统计（必须放在模块顶层才能被pickle）
def _count_chunk(
    args: tuple[str | Path, int, int, list[str] | None],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return countword(chunk, special_tokens)


# 并行统计：按 <|endoftext|> 切块，多进程各自统计后合并
def countword_parallel(
    input_path: str | Path,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
) -> Counter[tuple[bytes, ...]]:
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    # 用哪个special token作为切块边界（默认 <|endoftext|>）
    split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
    split_token_bytes = split_token.encode("utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    total: Counter[tuple[bytes, ...]] = Counter()
    if num_processes <= 1 or len(tasks) <= 1:
        for task in tasks:
            total.update(_count_chunk(task))
        return total

    with Pool(processes=num_processes) as pool:
        for partial in pool.imap_unordered(_count_chunk, tasks):
            total.update(partial)
    return total


# 主要训练
def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = _sorted_special_tokens(special_tokens)
    # 数词（并行预分词）
    word_freq = countword_parallel(input_path, special_tokens, num_processes)
    # 初始化
    vocab: dict[int, bytes] = {}
    next_id = 0
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[next_id] = token_bytes
        next_id += 1
    for b in range(256):
        vocab[next_id] = BYTE_TOKENS[b]
        next_id += 1

    if vocab_size <= len(vocab):
        return dict(list(vocab.items())[:vocab_size]), []
    # 准备训练的数据结构
    words: list[tuple[bytes, ...]] = list(word_freq.keys())
    freqs = [word_freq[word] for word in words]
    # 初始化pair统计
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_word_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    # 填充初始pair统计
    for word_id, symbols in enumerate(words):
        for pair, occurrences in _pair_occurrences(symbols).items():
            pair_counts[pair] += occurrences * freqs[word_id]
            pair_to_word_ids[pair].add(word_id)

    merges: list[tuple[bytes, bytes]] = []
    total_merges = vocab_size - len(vocab)

    # Lazy-deletion max-heap: (-count, _RevPair) so highest count wins,
    # with largest pair as tiebreak (matching the original max() behavior).
    heap: list = [(-count, _RevPair(pair)) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    while len(vocab) < vocab_size:
        # Pop stale heap entries until we find one whose count matches pair_counts.
        best_pair: tuple[bytes, bytes] | None = None
        while heap:
            neg_count, rev = heapq.heappop(heap)
            if pair_counts.get(rev.pair, 0) == -neg_count:
                best_pair = rev.pair
                break

        if best_pair is None:
            break

        merges.append(best_pair)
        vocab[next_id] = best_pair[0] + best_pair[1]
        next_id += 1
        if len(merges) % 500 == 0:
            print(f"  [2/2] 合并进度 {len(merges)}/{total_merges}", flush=True)

        affected_word_ids = list(pair_to_word_ids.pop(best_pair, ()))
        pair_counts.pop(best_pair, None)

        for word_id in affected_word_ids:
            old_symbols = words[word_id]
            word_freq_count = freqs[word_id]

            for pair, occurrences in _pair_occurrences(old_symbols).items():
                new_count = pair_counts.get(pair, 0) - occurrences * word_freq_count
                if new_count > 0:
                    pair_counts[pair] = new_count
                    heapq.heappush(heap, (-new_count, _RevPair(pair)))
                else:
                    pair_counts.pop(pair, None)
                word_ids = pair_to_word_ids.get(pair)
                if word_ids is not None:
                    word_ids.discard(word_id)
                    if not word_ids:
                        pair_to_word_ids.pop(pair, None)

            new_symbols = _merge_pair_in_symbols(old_symbols, best_pair)
            words[word_id] = new_symbols

            for pair, occurrences in _pair_occurrences(new_symbols).items():
                new_count = pair_counts.get(pair, 0) + occurrences * word_freq_count
                pair_counts[pair] = new_count
                pair_to_word_ids[pair].add(word_id)
                heapq.heappush(heap, (-new_count, _RevPair(pair)))

    return vocab, merges


def bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(
        input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens
    )


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.special_tokens = _sorted_special_tokens(special_tokens)

        existing_values = set(self.vocab.values())
        next_id = max(self.vocab.keys(), default=-1) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_values:
                self.vocab[next_id] = token_bytes
                existing_values.add(token_bytes)
                next_id += 1

        self.id_to_token = dict(self.vocab)
        self.token_to_id = {token: idx for idx, token in self.id_to_token.items()}
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_pattern = _compile_special_token_pattern(self.special_tokens)
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}
        for key, value in raw_vocab.items():
            vocab[int(key)] = (
                value.encode("latin1") if isinstance(value, str) else bytes(value)
            )

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                left, right = line.split(" ")
                merges.append((left.encode("latin1"), right.encode("latin1")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _encode_pretoken_bytes(self, token_bytes: bytes) -> tuple[int, ...]:
        cached = self._encode_cache.get(token_bytes)
        if cached is not None:
            return cached

        symbols = [BYTE_TOKENS[b] for b in token_bytes]
        while len(symbols) > 1:
            best_rank = None
            best_pair = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best_pair:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        encoded = tuple(self.token_to_id[symbol] for symbol in symbols)
        self._encode_cache[token_bytes] = encoded
        return encoded

    def _encode_ordinary_text(self, text: str) -> list[int]:
        encoded: list[int] = []
        for piece in pretokenizer(text):
            encoded.extend(self._encode_pretoken_bytes(piece.encode("utf-8")))
        return encoded

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self.special_pattern is None:
            return self._encode_ordinary_text(text)

        encoded: list[int] = []
        pos = 0
        for match in self.special_pattern.finditer(text):
            if match.start() > pos:
                encoded.extend(self._encode_ordinary_text(text[pos : match.start()]))
            encoded.append(self.token_to_id[match.group(0).encode("utf-8")])
            pos = match.end()

        if pos < len(text):
            encoded.extend(self._encode_ordinary_text(text[pos:]))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            last_newline = buffer.rfind("\n")
            if last_newline == -1:
                continue
            emit = buffer[: last_newline + 1]
            buffer = buffer[last_newline + 1 :]
            for token_id in self.encode(emit):
                yield token_id

        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.id_to_token[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")
