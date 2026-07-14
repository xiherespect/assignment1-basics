import json
import time
from pathlib import Path
from .bpe import train_bpe

if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "data" / "owt_train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print(f"[1/2] 预分词中...")
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path, vocab_size=vocab_size, special_tokens=special_tokens
    )
    print(
        f"完成！总耗时 {time.time() - t0:.1f}s，词汇表大小 {len(vocab)}，合并次数 {len(merges)}"
    )

    out = Path(__file__).parent.parent / "output"
    out.mkdir(exist_ok=True)
    json.dump(
        {k: v.decode("latin1") for k, v in vocab.items()},
        open(out / "vocab_owt.json", "w"),
        ensure_ascii=False,
    )
    open(out / "merges_owt.txt", "w").write(
        "\n".join(l.decode("latin1") + " " + r.decode("latin1") for l, r in merges)
    )
    print(f"已保存到 {out}/")

    longest = max(vocab.values(), key=len)
    print(f"最长 token ({len(longest)} bytes): {longest!r}")
