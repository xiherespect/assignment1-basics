import json
import time
from pathlib import Path

import numpy as np

from .bpe import train_bpe, BPETokenizer

# 文档分隔符 / 特殊 token。用 chr() 构造，避免源码里 <> 被渲染工具吞掉。
SPECIAL_TOKEN = chr(0x3C) + "|endoftext|" + chr(0x3E)


def train_bpe_by_data(data_name: str):
    input_path = Path(__file__).parent.parent / "data" / data_name
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    print(f"[1/2] 预分词中...")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path, vocab_size=vocab_size, special_tokens=special_tokens
    )
    print(
        f"完成！总耗时 {time.time() - t0:.1f}s，词汇表大小 {len(vocab)}，合并次数 {len(merges)}"
    )

    out = Path(__file__).parent.parent / "output" / data_name
    out.mkdir(exist_ok=True)
    json.dump(
        {k: v.decode("latin1") for k, v in vocab.items()},
        open(out / "vocab.json", "w"),
        ensure_ascii=False,
    )
    open(out / "merges.txt", "w").write(
        "\n".join(l.decode("latin1") + " " + r.decode("latin1") for l, r in merges)
    )
    print(f"已保存到 {out}/")

    longest = max(vocab.values(), key=len)
    print(f"最长 token ({len(longest)} bytes): {longest!r}")


def see_vocab():
    vocab_file = (
        Path(__file__).parent.parent / "output" / "owt_train.txt" / "vocab.json"
    )
    with open(vocab_file) as f:
        raw = json.load(f)
    vocab = {int(k): v.encode("latin1") for k, v in raw.items()}
    longest = max(vocab.values(), key=len)
    print(f"最长 token ({len(longest)} bytes): {longest!r}")


def _sample_documents(data_path: Path, num_docs: int = 10) -> list[str]:
    """按 <|endoftext|> 切分语料，取前 num_docs 个文档作为样本。"""
    docs: list[str] = []
    buffer = ""
    sep = "<|endoftext|>"
    with open(data_path, encoding="utf-8", errors="ignore") as f:
        while len(docs) < num_docs:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buffer += chunk
            while sep in buffer and len(docs) < num_docs:
                doc, buffer = buffer.split(sep, 1)
                if doc.strip():
                    docs.append(doc)
    return docs[:num_docs]


def compress(vocab_path, merges_path, data_path, num_docs: int | None = 20) -> float:
    """计算压缩比。num_docs=None 时流式跑全量语料（不占内存）。"""
    tokenizer = BPETokenizer.from_files(
        vocab_path, merges_path, special_tokens=["<|endoftext|>"]
    )
    data_path = Path(data_path)

    total_bytes = 0
    total_tokens = 0
    if num_docs is None:
        # 全量：字节数直接取文件大小，token 数用 encode_iterable 流式统计
        total_bytes = data_path.stat().st_size
        with open(data_path, encoding="utf-8", errors="ignore") as f:
            for _ in tokenizer.encode_iterable(f):
                total_tokens += 1
        label = "全量"
    else:
        docs = _sample_documents(data_path, num_docs)
        for doc in docs:
            total_bytes += len(doc.encode("utf-8"))
            total_tokens += len(tokenizer.encode(doc))
        label = f"文档数 {len(docs)}"

    ratio = total_bytes / total_tokens
    print(
        f"{data_path}: {label}，总字节 {total_bytes}，"
        f"总 token {total_tokens}，压缩比 {ratio:.2f} bytes/token"
    )
    return ratio


_NPY_MAGIC = b"\x93NUMPY"
_NPY_HEADER_BLOCK = 128  # 固定 header 大小，足够放下 shape(~20 位数字) 并对齐到 16


def _npy_header(n: int, dtype_str: str = "<u2") -> bytes:
    """构造固定 128 字节的 npy v1 header，shape=(n,)。"""
    version = b"\x01\x00"
    inner_len = _NPY_HEADER_BLOCK - len(_NPY_MAGIC) - len(version) - 2  # 118
    inner = (
        "{'descr': '"
        + dtype_str
        + "', 'fortran_order': False, 'shape': ("
        + str(n)
        + ",)}"
    )
    if len(inner) + 1 > inner_len:
        raise ValueError(
            f"npy header 过大（{len(inner)+1} > {inner_len}），请增大 _NPY_HEADER_BLOCK"
        )
    inner = inner + " " * (inner_len - len(inner) - 1) + "\n"
    return _NPY_MAGIC + version + len(inner).to_bytes(2, "little") + inner.encode("latin1")


def encode_to_npy(
    vocab_path,
    merges_path,
    data_path,
    out_path,
    special_tokens: list[str] | None = None,
) -> int:
    """把语料编码成 token ID 序列，存成 uint16 的 .npy（单遍流式，无临时文件）。

    用 encode_iterable 边读边编码，分块写入裸 uint16 字节，最后回填 header 里的
    token 总数。词表大小 < 65536，所以 uint16 足够且最省空间。
    """
    if special_tokens is None:
        special_tokens = [SPECIAL_TOKEN]
    tokenizer = BPETokenizer.from_files(
        vocab_path, merges_path, special_tokens=special_tokens
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_id = max(tokenizer.vocab.keys())
    if max_id >= 2**16:
        raise ValueError(
            f"词表最大 ID {max_id} 超出 uint16 范围(0~65535)，请改用 uint32"
        )

    # 先写固定大小 header 占位，随后流式追加裸数据，最后回填真实 token 数
    buf: list[int] = []
    chunk = 1 << 20  # 每攒 ~1M 个 token flush 一次
    total = 0
    with open(out_path, "wb") as f:
        f.write(_npy_header(0))  # 占位

        with open(data_path, encoding="utf-8", errors="ignore") as src:
            for token_id in tokenizer.encode_iterable(src):
                buf.append(token_id)
                if len(buf) >= chunk:
                    f.write(np.array(buf, dtype=np.uint16).tobytes())
                    total += len(buf)
                    buf.clear()
        if buf:
            f.write(np.array(buf, dtype=np.uint16).tobytes())
            total += len(buf)

        f.seek(0)
        f.write(_npy_header(total))

    print(f"{data_path} -> {out_path}: {total} tokens (uint16)")
    return total


def encode_all() -> None:
    """把 TinyStories / OpenWebText 各自的 train+valid 编码成 4 个 uint16 .npy。

    输出布局：
      output/tokenized/tinystories/train.tokens.npy
      output/tokenized/tinystories/valid.tokens.npy
      output/tokenized/owt/train.tokens.npy
      output/tokenized/owt/valid.tokens.npy
    """
    root = Path(__file__).parent.parent
    out_root = root / "output" / "tokenized"

    jobs = [
        # (名称, vocab, merges, 数据, 输出)
        (
            "TinyStories-train",
            root / "output" / "vocab_tiny.json",
            root / "output" / "merges_tiny.txt",
            root / "data" / "TinyStoriesV2-GPT4-train.txt",
            out_root / "tinystories" / "train.tokens.npy",
        ),
        (
            "TinyStories-valid",
            root / "output" / "vocab_tiny.json",
            root / "output" / "merges_tiny.txt",
            root / "data" / "TinyStoriesV2-GPT4-valid.txt",
            out_root / "tinystories" / "valid.tokens.npy",
        ),
        (
            "OWT-train",
            root / "output" / "owt_train.txt" / "vocab.json",
            root / "output" / "owt_train.txt" / "merges.txt",
            root / "data" / "owt_train.txt",
            out_root / "owt" / "train.tokens.npy",
        ),
        (
            "OWT-valid",
            root / "output" / "owt_train.txt" / "vocab.json",
            root / "output" / "owt_train.txt" / "merges.txt",
            root / "data" / "owt_valid.txt",
            out_root / "owt" / "valid.tokens.npy",
        ),
    ]

    for name, vocab, merges, data, out in jobs:
        if not data.exists():
            print(f"[skip] {name}: 找不到数据 {data}")
            continue
        print(f"=== {name} ===")
        t0 = time.time()
        encode_to_npy(vocab, merges, data, out)
        print(f"  耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    encode_all()
