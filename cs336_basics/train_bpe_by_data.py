import json
import time
from pathlib import Path
from .bpe import train_bpe


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
    vocab_file = Path(__file__).parent.parent / "output" / "vocab_tiny.json"
    with open(vocab_file) as f:
        raw = json.load(f)
    vocab = {int(k): v.encode("latin1") for k, v in raw.items()}
    longest = max(vocab.values(), key=len)
    print(f"最长 token ({len(longest)} bytes): {longest!r}")


if __name__ == "__main__":
    # train_bpe_by_data("TinyStoriesV2-GPT4-train.txt")
    see_vocab()
