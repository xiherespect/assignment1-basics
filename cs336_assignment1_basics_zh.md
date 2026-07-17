# CS336 作业 1（基础）：构建 Transformer 语言模型

**版本 26.0.3**
**CS336 教学团队**
**2026 年春季**

---

## 1 作业概述

在本作业中，你将从头构建训练一个标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

### 你将实现的内容

1. 字节对编码（BPE）分词器（第 2 节）
2. Transformer 语言模型（LM）（第 3 节）
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）
4. 训练循环，支持模型和优化器状态的序列化与加载（第 5 节）

### 你将运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行采样生成并评估困惑度。
5. 在 OpenWebText 上训练模型，并将你获得的困惑度提交到排行榜。

### 你可以使用的内容

我们期望你从头构建每个组件。特别地，你**不可以**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义，但以下情况除外：

* `torch.nn.Parameter`
* `torch.nn` 中的容器类（如 `Module`、`ModuleList`、`Sequential` 等）。¹
* `torch.optim.Optimizer` 基类

你可以使用任何其他的 PyTorch 定义。如果你想使用某个函数或类但不确定是否被允许，欢迎在 Slack 上询问。当有疑问时，请考虑使用它是否会破坏本作业"从头构建"的精神。

### 关于 AI 工具的声明

AI 可以完全自主地解决作业的许多部分。这会使你更难深入地参与并从课程材料中学习。

允许使用 AI 工具来回答高层次的概念性问题，或提供底层的编程文档，如函数签名和库 API。然而，**不允许**使用 AI 工具来实现作业的任何部分。这既包括编程代理（如 Cursor Agents、Codex、Claude Code），也包括 AI 自动补全（如 Cursor Tab、GitHub Copilot）。当使用 AI 代理时，请确保它使用提供的 AGENTS.md 文件。使用聊天机器人时也应包含该 prompt。

我们强烈建议你在完成作业时禁用 IDE 中的 AI 自动补全（如 Cursor Tab、GitHub Copilot）（不过非 AI 自动补全，如函数名自动补全完全可以使用）。往届学生强调，禁用 AI 自动补全使他们更容易深入理解课程内容。

完整的 AI 政策请参阅相关文档。

### 代码结构

作业代码和本说明文档可在 GitHub 上获取：

> `github.com/stanford-cs336/assignment1-basics`

请 `git clone` 该仓库。如有任何更新，我们会通知你进行 `git pull` 获取最新版本。

1. **`cs336_basics/*`**：这是你编写代码的地方。注意这里没有任何现成代码——你可以从头开始任意实现！
2. **`adapters.py`**：你的代码必须具备一组功能。对于每项功能（如缩放点积注意力），通过简单地调用你的实现来填写其测试适配器（如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；这只是胶水代码。
3. **`test_*.py`**：包含你必须通过的所有测试（如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中定义的钩子。不要编辑测试文件。

### 如何提交

为了提交，请运行 `make_submission.sh` 来构造一个提交 zip 文件。如果你有大型数据文件或检查点不想包含在提交 zip 中，请确保将它们添加到脚本的排除列表中。

你将向 Gradescope 提交以下文件：

* **`writeup.pdf`**：回答所有书面问题。请对你的回答进行排版。
* **`code.zip`**：包含你编写的所有代码。

要提交到排行榜，请向以下仓库提交一个 PR：

> `github.com/stanford-cs336/assignment1-basics-leaderboard`

详细的提交说明请参阅排行榜仓库中的 `README.md`。

### 从哪里获取数据集

本作业将使用两个预处理数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两个数据集都是单个大型纯文本文件。

如果你随课程一起完成作业，可以在计算指南中找到下载数据集的说明。

如果你在家跟随学习，可以使用 `README.md` 中的命令下载这些文件。

> **低资源提示：初始化**
>
> 在整个课程的作业讲义中，我们会给出一些建议，帮助你在较少或没有 GPU 资源的情况下完成作业的某些部分。例如，我们有时会建议对数据集或模型规模进行**降采样（downscaling）**，或者解释如何在配备集成 GPU 或 CPU 的 Mac 上运行训练代码。你会在蓝色框（就像这个）中找到这些"低资源提示"。即使你是能够访问课程机器的 Stanford 在校生，这些提示也可能帮助你更快地迭代、节省时间，所以我们建议阅读它们！

> **低资源提示：在 Apple Silicon 或 CPU 上完成作业 1**
>
> 使用教学团队的参考代码，我们可以在配备 36 GB RAM 的 Apple M4 Max 芯片上，使用 Metal GPU（MPS）在 5 分钟内、使用 CPU 约 30 分钟内，训练一个能生成合理流畅文本的 LM。如果这些术语对你来说没什么意义，不用担心！只需知道，如果你有一台较新的笔记本电脑，并且实现正确且高效，你就能训练一个小型 LM，生成具有不错流畅度的简单儿童故事。
>
> 在作业后面部分，我们将解释如果你在 CPU 或 MPS 上需要做哪些调整。

> ¹ 完整列表见 `pytorch.org/docs/stable/nn.html#containers`。

---

## 2 字节对编码（BPE）分词器

在作业的第一部分，我们将训练并实现一个字节级字节对编码（BPE）分词器 [R. Sennrich et al., 2016；C. Wang et al., 2019]。特别地，我们会将任意（Unicode）字符串表示为字节序列，并在此字节序列上训练 BPE 分词器。之后，我们将使用这个分词器把文本（字符串）编码为 token（整数序列），用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它将字符映射到整数**码点（code points）**。截至 Unicode 17.0（于 2025 年 9 月发布），该标准定义了 172 种书写系统中的 159,801 个字符。例如，字符 "s" 的码点是 115（通常记作 `U+0073`，其中 `U+` 是约定的前缀，`0073` 是十六进制的 115），字符 "牛" 的码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数将整数 Unicode 码点转换为对应的字符字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

> **问题 (unicode1)：理解 Unicode（1 分）**
>
> **(a)** `chr(0)` 返回什么 Unicode 字符？
>
> **交付物**：一句话回答。
>
> **(b)** 这个字符的字符串表示（`__repr__()`）与其打印表示有何不同？
>
> **交付物**：一句话回答。
>
> **(c)** 当这个字符出现在文本中时会发生什么？在你的 Python 解释器中尝试以下内容，看看是否符合你的预期可能会有帮助：
>
> ```python
> >>> chr(0)
> >>> print(chr(0))
> >>> "this is a test" + chr(0) + "string"
> >>> print("this is a test" + chr(0) + "string")
> ```
>
> **交付物**：一句话回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器并不实际，因为词汇表会大得难以承受（约 15 万项）且稀疏（因为许多字符相当罕见）。因此，我们将使用一种 Unicode **编码**，它将 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主导编码（超过 98% 的网页）。

要将 Unicode 字符串编码为 UTF-8，我们可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值，我们可以对其进行迭代（如调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节串解码为 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码后字符串的字节值（0 到 255 的整数）。
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227,
129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们实际上是把码点序列（21 位整数，有 159,801 个有效值）转换为字节值序列（0 到 255 范围内的整数）。这个长度为 256 的字节词汇表**处理起来容易得多**。使用字节级分词时，我们无需担心词汇表外（out-of-vocabulary）的 token，因为我们知道**任何**输入文本都可以表示为 0 到 255 的整数序列。

> **问题 (unicode2)：Unicode 编码（3 分）**
>
> **(a)** 相比 UTF-16 或 UTF-32，为什么我们更倾向于在 UTF-8 编码的字节上训练分词器？比较这些编码对各种输入字符串的输出可能会有帮助。
>
> **交付物**：一到两句话回答。
>
> **(b)** 考虑以下（不正确的）函数，它本意是将 UTF-8 字节串解码为 Unicode 字符串。为什么这个函数不正确？给出一个产生不正确结果的输入字节串示例。
>
> ```python
> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
>     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
> 'hello'
> ```
>
> **交付物**：一个使 `decode_utf8_bytes_to_str_wrong` 产生不正确输出的输入字节串示例，并用一句话解释为什么该函数不正确。
>
> **(c)** 给出一个不能解码为任何 Unicode 字符的双字节序列。
>
> **交付物**：一个示例，并用一句话解释。

### 2.3 子词分词

虽然字节级分词可以缓解词级分词器面临的词汇表外问题，但将文本分词为字节会产生极长的序列。这会减慢模型训练速度，因为一个 10 个单词的句子在词级语言模型中可能只有 10 个 token，但在字符级模型中可能有 50 个或更多 token（取决于单词长度）。处理这些更长的序列需要在模型的每一步进行更多计算。此外，在字节序列上进行语言建模是困难的，因为更长的输入序列会在数据中产生长期依赖。

**子词分词**是词级分词器和字节级分词器之间的折中。注意，字节级分词器的词汇表有 256 个条目（字节值 0 到 255）。子词分词器以更大的词汇表规模换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中频繁出现，为它分配一个词汇表条目会将这个 3-token 序列缩减为单个 token。

我们如何选择要添加到词汇表的这些子词单元？[R. Sennrich et al. [3]] 提出使用字节对编码（BPE；[P. Gage [5]]），这是一种压缩算法，它迭代地将最频繁的字节对替换（"合并"）为一个新的、未使用的索引。注意，该算法会向词汇表添加子词 token，以最大化对我们输入序列的压缩——如果一个单词在输入文本中出现足够多次，它将被表示为单个子词单元。

使用通过 BPE 构造的词汇表的子词分词器通常被称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其中词汇表项是字节或合并后的字节序列，这给了我们在词汇表外处理和可管理的输入序列长度两方面的优点。构造 BPE 分词器词汇表的过程被称为"训练" BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器的训练过程由三个主要步骤组成。

**词汇表初始化**

分词器词汇表是从字节串 token 到整数 ID 的一对一映射。由于我们训练的是字节级 BPE 分词器，我们的初始词汇表就是所有字节的集合。由于有 256 个可能的字节值，我们的初始词汇表大小为 256。

**预分词**

一旦有了词汇表，原则上你可以统计文本中字节彼此相邻出现的频率，并从最频繁的字节对开始合并。然而，这在计算上相当昂贵，因为我们每次合并都必须对语料库进行一次完整遍历。此外，直接跨语料库合并字节可能会产生仅在标点上不同的 token（如 `dog!` vs. `dog.`）。尽管这些 token 可能在语义上非常相似（仅在标点上不同），它们会得到完全不同的 token ID。

为避免这一点，我们对语料库进行**预分词**。你可以将其视为对语料库的一种粗粒度分词，帮助我们统计字符对出现的频率。例如，单词 `'text'` 可能是一个出现了 10 次的预 token。在这种情况下，当我们统计字符 't' 和 'e' 相邻出现的频率时，我们会看到单词 'text' 中 't' 和 'e' 相邻，并可以把它们的计数增加 10，而不是遍历整个语料库。由于我们训练的是字节级 BPE 模型，每个预 token 被表示为一个 UTF-8 字节序列。

[R. Sennrich et al. [3]] 的原始 BPE 实现通过简单地按空白分割来预分词（即 `s.split(" ")`）。这种方法在基于 SentencePiece 的分词器中仍然可见（例如 Llama 1 和 2 分词器）。

大多数现代分词器使用基于正则表达式的预分词器，这是来自 GPT-2 [A. Radford et al. [6]] 的做法。我们将使用原始正则表达式的稍微更美观的形式，取自 `github.com/openai/tiktoken/pull/234/files`：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个预分词器分割一些文本，可以帮助你更好地理解它的行为：

```python
>>> # 需要 `regex` 包
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过，在你的代码中使用它时，你应该使用 `re.finditer`，以避免在构造从预 token 到其计数的映射时存储预分词后的单词。

**计算 BPE 合并**

现在我们已经将输入文本转换为预 token，并将每个预 token 表示为 UTF-8 字节序列，我们可以计算 BPE 合并（即训练 BPE 分词器）。从高层次看，BPE 算法迭代地统计每一对字节，并找出频率最高的字节对（"A"，"B"）。这个最频繁字节对（"A"，"B"）的每次出现都会被**合并**，即被替换为一个新 token "AB"。这个新的合并 token 被添加到我们的词汇表；因此，BPE 训练后的最终词汇表大小为初始词汇表大小（在我们的例子中是 256），加上训练期间执行的 BPE 合并操作次数。为了在 BPE 训练期间提高效率，我们不考虑跨预 token 边界的字节对。² 在计算合并时，我们通过**优先选择字典序更大的字节对**来确定性地打破频率上的平局。例如，如果字节对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都有最高频率，我们会合并 ("BA", "A")：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**特殊 token**

通常，一些字符串（如 `<|endoftext|>`）用于编码元数据（如文档之间的边界）。在编码文本时，通常希望将某些字符串视为"特殊 token"，它们绝不应被拆分为多个 token（即始终作为单个 token 保留）。例如，序列结束字符串 `<|endoftext|>` 应始终作为单个 token 保留（即单个整数 ID），以便我们知道何时停止从语言模型生成。这些特殊 token 必须被添加到词汇表，以便它们有对应的固定 token ID。

[R. Sennrich et al. [3]] 的算法 1 包含 BPE 分词器训练的一个低效实现（本质上遵循我们上面概述的步骤）。作为第一个练习，实现并测试这个函数以检查你的理解可能会有帮助。

> **示例 (bpe_example)：BPE 训练示例**
>
> 这是来自 [R. Sennrich et al. [3]] 的一个风格化示例。考虑一个由以下文本组成的语料库：
>
> ```
> low low low low low
> lower lower widest widest widest
> newest newest newest newest newest newest
> ```
>
> 词汇表有一个特殊 token `<|endoftext|>`。
>
> **词汇表**
>
> 我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词汇表。
>
> **预分词**
>
> 为简单起见并专注于合并过程，本示例中我们假设预分词只是按空白分割。当我们预分词并统计时，最终得到如下频率表。
>
> ```
> {low: 5, lower: 2, widest: 3, newest: 6}
> ```
>
> 将其表示为 `dict[tuple[bytes, ...], int]` 会很方便，例如 `{(l,o,w): 5, …}`。注意，即使是单个字节在 Python 中也是 `bytes` 对象。Python 中没有 `byte` 类型来表示单个字节，正如没有 `char` 类型来表示单个字符一样。
>
> **合并**
>
> 我们首先查看每一对连续字节，并对这些单词出现的频率求和，得到 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。字节对 ('e', 's') 和 ('s', 't') 平局，所以我们取字典序更大的字节对 ('s', 't')。然后我们合并这些预 token，最终得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。
>
> 在第二轮，我们看到 (e, st) 是最常见的字节对（计数为 9），我们会合并为 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续这样做，我们最终得到的合并序列是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。
>
> 如果我们取 6 次合并，我们有 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词汇表元素为 `[<|endoftext|>, [...256 个字节字符], st, est, ow, low, west, ne]`。
>
> 使用这个词汇表和合并集，单词 `newest` 会被分词为 `[ne, west]`。

> ² 注意，原始 BPE 表述 [R. Sennrich et al. [3]] 指定了包含一个词尾 token。我们在训练字节级 BPE 模型时不添加词尾 token，因为所有字节（包括空格和标点）都包含在模型的词汇表中。由于我们显式地表示空格和标点，学到的 BPE 合并将自然反映这些单词边界。

### 2.5 BPE 分词器训练实践

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明可在第 1 节中找到。开始之前，我们建议看一看 TinyStories 数据集，以了解数据中的内容。

**并行化预分词**

你会发现一个主要瓶颈是预分词步骤。你可以通过使用内置库 `multiprocessing` 并行化代码来加速预分词。具体来说，我们建议在预分词的并行实现中对语料库分块，同时确保你的分块边界出现在特殊 token 的开头。你可以逐字使用以下链接中的起始代码来获取分块边界，然后用它们在你的进程之间分配工作：

> `https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

这种分块始终是有效的，因为我们绝不会跨文档边界合并。对于本作业的目的，你总是可以这样分割。不用担心接收到一个不包含 `<|endoftext|>` 的超大语料库这种边界情况。

**在预分词前移除特殊 token**

在使用正则表达式模式运行预分词之前（使用 `re.finditer`），你应该剔除语料库（或你的分块，如果是并行实现）中的所有特殊 token。确保你在特殊 token 上**分割**，以便不会跨它们所界定的文本进行合并。例如，如果你有一个语料库（或分块）像 `[Doc 1]<|endoftext|>[Doc 2]`，你应该在特殊 token `<|endoftext|>` 上分割，得到 `[Doc 1]` 和 `[Doc 2]` 并分别预分词，以便不会跨文档边界发生合并。换句话说，特殊 token 在训练期间定义硬分割边界，但它们本身不应贡献于合并计数。这可以使用 `re.split` 完成，以 `"|".join(special_tokens)` 作为分隔符（小心使用 `re.escape`，因为 `|` 可能出现在特殊 token 中）。测试 `test_train_bpe_special_tokens` 会对此进行测试。

**优化合并步骤**

上面风格化示例中的朴素 BPE 训练实现很慢，因为对于每次合并，它都会遍历所有字节对以找出最频繁的字节对。然而，每次合并后唯一发生变化的字节对计数是那些与被合并字节对重叠的字节对。因此，BPE 训练速度可以通过对所有字节对的计数建立索引并增量更新这些计数来提高，而不是显式遍历每一对字节来统计字节对频率。使用这种缓存过程可以获得显著的加速，不过我们注意到 BPE 训练的合并部分在 Python 中是**不可**并行化的。

> **低资源提示：性能分析（Profiling）**
>
> 你应该使用 `cProfile` 或 `py-spy` 等性能分析工具来识别你实现中的瓶颈，并专注于优化它们。

> **低资源提示："降采样（Downscaling）"**
>
> 我们建议你不要一上来就在完整的 TinyStories 数据集上训练分词器，而是先在数据的一个小子集上训练：一个"调试数据集"。例如，你可以在 TinyStories 验证集上训练分词器，它有 22K 个文档而不是 2.12M 个。这说明了一个在可能时尽量降采样以加速开发的一般策略：例如，使用更小的数据集、更小的模型规模等。选择调试数据集或超参数配置的规模需要仔细考虑：你希望调试集足够大，以具有与完整配置相同的瓶颈（这样你所做的优化才能泛化），但又不能太大以致于运行起来花费很长时间。

> **问题 (train_bpe)：BPE 分词器训练（15 分）**
>
> **交付物**：编写一个函数，给定一个输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应（至少）处理以下输入参数：
>
> * **`input_path: str`** 包含 BPE 分词器训练数据的文本文件路径。
> * **`vocab_size: int`** 一个正整数，定义最终词汇表的最大大小（包括初始字节词汇表、合并产生的词汇表项以及任何特殊 token）。
> * **`special_tokens: list[str]`** 要添加到词汇表的字符串列表。在训练期间，将它们视为防止跨其跨度合并的硬边界，但在计算合并统计时不包括它们。
>
> 你的 BPE 训练函数应返回结果词汇表和合并：
>
> * **`vocab: dict[int, bytes]`** 分词器词汇表，从 `int`（词汇表中的 token ID）到 `bytes`（token 字节）的映射。
> * **`merges: list[tuple[bytes, bytes]]`** 训练产生的 BPE 合并列表。每个列表项是一个字节元组 (`<token1>`, `<token2>`)，表示 `<token1>` 与 `<token2>` 合并。合并应按创建顺序排序。
>
> 要针对我们提供的测试来测试你的 BPE 训练函数，你首先需要实现测试适配器 **[adapters.run_train_bpe]**。然后，运行 `uv run pytest tests/test_train_bpe.py`。你的实现应能通过所有测试。可选地（这可能是一项很大的时间投入），你可以使用某种系统语言实现训练方法的关键部分，例如 C++（考虑 `cppyy` 或 `nanobind`）或 Rust（使用 PyO3）。如果你这样做，请注意哪些操作需要从 Python 内存复制而不是直接读取，并确保留下构建说明，或确保仅使用 `pyproject.toml` 即可构建。还要注意，GPT-2 正则表达式在大多数正则引擎中支持不佳，并且在大多数支持它的引擎中会太慢。我们已经验证 Oniguruma 相当快并支持负向先行断言，但 Python 中的 `regex` 包如果说有什么不同的话，甚至更快。

> **问题 (train_bpe_tinystories)：在 TinyStories 上训练 BPE（2 分）**
>
> **(a)** 在 TinyStories 数据集上训练一个字节级 BPE 分词器，使用最大词汇表大小 10,000。确保将 TinyStories `<|endoftext|>` 特殊 token 添加到词汇表。将结果词汇表和合并序列化到磁盘以供进一步检查。训练花费了多少时间和内存？词汇表中最长的 token 是什么？它有意义吗？
>
> **资源要求**：≤ 30 分钟（无 GPU），≤ 30 GB RAM
>
> **提示** 你应该能够在预分词期间使用 `multiprocessing` 并利用以下两个事实，使 BPE 训练时间控制在 2 分钟以内：
>
> (a) `<|endoftext|>` token 在数据文件中界定文档。
> (b) `<|endoftext|>` token 在应用 BPE 合并之前作为特殊情况处理。
>
> **交付物**：一到两句话回答。
>
> **(b)** 分析你的代码。分词器训练过程的哪一部分花费的时间最多？
>
> **交付物**：一到两句话回答。

接下来，我们将尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。和之前一样，我们建议看一看数据集以更好地理解其内容。

> **问题 (train_bpe_expts_owt)：在 OpenWebText 上训练 BPE（2 分）**
>
> **(a)** 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，使用最大词汇表大小 32,000。将结果词汇表和合并序列化到磁盘以供进一步检查。词汇表中最长的 token 是什么？它有意义吗？
>
> **资源要求**：≤ 12 小时（无 GPU），≤ 100 GB RAM
>
> **交付物**：一到两句话回答。
>
> **(b)** 对比你在 TinyStories 与 OpenWebText 上训练得到的分词器。
>
> **交付物**：一到两句话回答。

### 2.6 BPE 分词器：编码与解码

在作业的上一部分，我们实现了一个在输入文本上训练 BPE 分词器以获得分词器词汇表和 BPE 合并列表的函数。现在，我们将实现一个 BPE 分词器，它加载提供的词汇表和合并列表，并用它们把文本编码为 token ID 和从 token ID 解码。

#### 2.6.1 编码文本

用 BPE 编码文本的过程镜像了我们训练 BPE 词汇表的方式。有几个主要步骤。

**步骤 1：预分词。** 我们首先预分词序列，并将每个预 token 表示为 UTF-8 字节序列，就像我们在 BPE 训练中所做的那样。我们将在每个预 token 内把这些字节合并为词汇表元素，独立处理每个预 token（不跨预 token 边界合并）。

**步骤 2：应用合并。** 然后我们取 BPE 训练期间创建的词汇表元素合并序列，并**按创建顺序**将其应用于我们的预 token。

> **示例 (bpe_encoding)：BPE 编码示例**
>
> 例如，假设我们的输入字符串是 `'the cat ate'`，我们的词汇表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，我们学到的合并是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，我们的预分词器会将这个字符串分割为 `['the', ' cat', ' ate']`。然后，我们查看每个预 token 并应用 BPE 合并。
>
> 第一个预 token `'the'` 最初被表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们识别出第一个适用的合并是 `(b't', b'h')`，用它将预 token 转换为 `[b'th', b'e']`。然后，我们回到合并列表并识别下一个适用的合并 `(b'th', b'e')`，它将预 token 转换为 `[b'the']`。最后，回看合并列表，我们发现没有更多适用于该字符串的合并（因为整个预 token 已被合并为单个 token），所以我们完成了 BPE 合并的应用。对应的整数序列是 `[9]`。
>
> 对剩余的预 token 重复这个过程，我们看到预 token `' cat'` 在应用 BPE 合并后被表示为 `[b' c', b'a', b't']`，它变成整数序列 `[7, 1, 5]`。最后的预 token `' ate'` 在应用 BPE 合并后是 `[b' at', b'e']`，它变成整数序列 `[10, 3]`。因此，编码我们输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊 token**

你的分词器在编码文本时应能正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考虑**

假设我们想要分词一个无法放入内存的大型文本文件。为了高效地分词这个大文件（或任何其他数据流），我们需要把它拆分为可管理的块并依次处理每个块，使内存复杂度保持恒定而非与文本大小成线性。这样做时，我们需要确保一个 token 不跨越块边界，否则我们会得到与在内存中朴素地分词整个序列不同的分词结果。

#### 2.6.2 解码文本

要将 token ID 序列解码回原始文本，我们可以简单地在词汇表中查找每个 ID 对应的条目（一个字节序列），把它们连接在一起，然后将字节解码为 Unicode 字符串。注意，输入 ID 不保证映射到有效的 Unicode 字符串（因为用户可以输入任意整数 ID 序列）。如果输入 token ID 不产生有效的 Unicode 字符串，你应该用官方 Unicode 替换字符 `U+FFFD` 替换格式错误的字节。³ `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 会自动用替换标记替换格式错误的数据。

> **问题 (tokenizer)：实现分词器（15 分）**
>
> **交付物**：实现一个 `Tokenizer` 类，给定一个词汇表和一个合并列表，将文本编码为整数 ID 并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果它们不在词汇表中，则将其追加到词汇表）。我们推荐以下接口：
>
> **`def __init__(self, vocab, merges, special_tokens=None)`** 从给定的词汇表、合并列表和（可选的）特殊 token 列表构造分词器。此函数应接受以下参数：
>
> * `vocab: dict[int, bytes]`
> * `merges: list[tuple[bytes, bytes]]`
> * `special_tokens: list[str] | None = None`
>
> **`def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`** 类方法，从序列化的词汇表和合并列表（与你的 BPE 训练代码输出格式相同）以及（可选的）特殊 token 列表构造并返回一个 `Tokenizer`。此方法应接受以下额外参数：
>
> * `vocab_filepath: str`
> * `merges_filepath: str`
> * `special_tokens: list[str] | None = None`
>
> **`def encode(self, text: str) -> list[int]`** 将输入文本编码为 token ID 序列。
>
> **`def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`** 给定一个字符串可迭代对象（如 Python 文件句柄），返回一个惰性产生 token ID 的生成器。这对于我们无法直接加载到内存的大文件的内存高效分词是必需的。
>
> **`def decode(self, ids: list[int]) -> str`** 将 token ID 序列解码为文本。
>
> 要针对我们提供的测试来测试你的 `Tokenizer`，你首先需要实现测试适配器 **[adapters.get_tokenizer]**。然后，运行 `uv run pytest tests/test_tokenizer.py`。你的实现应能通过所有测试。

> ³ 关于 Unicode 替换字符的更多信息，见 `en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character`。

### 2.7 实验

> **问题 (tokenizer_experiments)：分词器实验（4 分）**
>
> **(a)** 从 TinyStories 和 OpenWebText 各采样 10 个文档。使用你之前训练的 TinyStories 和 OpenWebText 分词器（分别为 10K 和 32K 词汇表大小），将这些采样的文档编码为整数 ID。每个分词器的压缩比（字节/token）是多少？
>
> **交付物**：/u01/lvliping/src/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt: 文档数 20，总字节 18259，总 token 4481，压缩比 4.07 bytes/token
> /u01/lvliping/src/assignment1-basics/data/owt_valid.txt: 文档数 20，总字节 91351，总 token 19970，压缩比 4.57 bytes/token
>
> **(b)** 如果你用 TinyStories 分词器分词 OpenWebText 样本会发生什么？比较压缩比和/或定性地描述会发生什么。
>
> **交付物** 压缩比会下降，变成3.3
>
> **(c)** 估计你的分词器的吞吐量（如字节/秒）。分词 Pile 数据集（825GB 文本）需要多长时间？
>
> **交付物**：吞吐量 ≈ 3.9 MB/s(由 290MB / 74.4s 实测得到)
分词 Pile(825GB) ≈ 825 GB ÷ 3.9 MB/s ≈ 2.3×10⁵ 秒 ≈ 63 小时 ≈ 2.6 天(单进程纯 Python)
>
> **(d)** 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练集和开发集编码为整数 token ID 序列。我们稍后会用它来训练语言模型。我们建议将 token ID 序列化为数据类型为 `uint16` 的 NumPy 数组。为什么 `uint16` 是一个合适的选择？
>
> **交付物**：因为uint16可以装得下32k的词表
---

## 3 Transformer 语言模型架构

语言模型接受一个批量的整数 token ID 序列作为输入（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回一个（批量的）在词汇表上归一化的概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中预测的分布是对每个输入 token 的下一个词的分布。训练语言模型时，我们使用这些下一个词的预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理期间从语言模型生成文本时，我们取最后一个时间步的预测下一个词分布（即序列中的最后一项），生成序列中的下一个 token（例如取概率最高的 token、从分布中采样等），将生成的 token 添加到输入序列，然后重复。

在作业的这一部分，你将从头构建这个 Transformer 语言模型。我们将从模型的高层次描述开始，然后逐步详述各个组件。

### 3.1 Transformer LM

给定一个 token ID 序列，Transformer 语言模型使用输入嵌入将 token ID 转换为稠密向量，将嵌入后的 token 通过 `num_layers` 个 Transformer 块，然后应用一个学习到的线性投影（"输出嵌入"或"LM 头"）来产生预测的下一个 token logits。图 1 为示意图表示。

**图 1**：我们 Transformer 语言模型的概览。（自底向上：Inputs → Token Embedding → Transformer Block × num_layers → Norm → Linear（输出嵌入）→ Softmax → Output Probabilities）

**图 2**：一个 pre-norm Transformer 块。（输入张量形状 `(batch_size, seq_len, d_model)` → Norm → Causal Multi-Head Self-Attention w/ RoPE → Add（残差）→ Norm → Position-Wise Feed-Forward → Add（残差）→ 输出张量形状 `(batch_size, seq_len, d_model)`）

**Token 嵌入**

在最开始一步，Transformer 将（批量的）token ID 序列**嵌入**为一个向量序列，这些向量包含关于 token 身份的信息（图 1 中的红色块）。

更具体地说，给定一个 token ID 序列，Transformer 语言模型使用一个 token 嵌入层来产生一个向量序列。每个嵌入层接受一个形状为 `(batch_size, sequence_length)` 的整数张量，并产生一个形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

**Pre-norm Transformer 块**

嵌入之后，激活由若干个结构相同的神经网络层处理。一个标准的仅解码器（decoder-only）Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer"块"）组成。每个 Transformer 块接受一个形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回一个形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块通过自注意力在序列间聚合信息，并通过前馈层对其进行非线性变换。

在 `num_layers` 个 Transformer 块之后，我们将最终激活转换为词汇表上的分布。

我们将实现"pre-norm" Transformer 块（详见第 3.4 节），它额外要求在最后一个 Transformer 块之后使用层归一化（下面详述），以确保其输出被正确缩放。

在此归一化之后，我们将使用一个标准的学习到的线性变换，将 Transformer 块的输出转换为预测的下一个 token logits（参见 [A. Radford et al. [7]] 的公式 2）。

### 3.2 补充说明：批处理、Einsum 与高效计算

在整个 Transformer 中，我们将对许多批量式的输入执行相同的计算。以下是几个例子：

* **一个批次的元素**：我们对每个批次元素应用相同的 Transformer `forward` 操作。
* **序列长度**：像 RMSNorm 和前馈这样的"逐位置（position-wise）"操作在序列的每个位置上相同地进行。
* **注意力头**：在"多头"注意力操作中，注意力操作在多个注意力头上进行批处理。

以一种能够充分利用 GPU、又易于阅读和理解的方式来执行这些操作是很有用的。许多 PyTorch 操作可以在张量的开头接受多余的"批量式"维度，并高效地在这些维度上重复/广播操作。

例如，假设我们进行一个逐位置的批量操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量" $D$，我们想对一个形状为 `(d_model, d_model)` 的矩阵 $A$ 进行批量的向量-矩阵乘法。在这种情况下，`D @ A` 会进行批量矩阵乘法，其中 `(batch_size, sequence_length)` 维度被批处理。

因此，假设你的函数可能被给予额外的批量式维度，并将这些维度保持在 PyTorch 形状的开头，会很有帮助。为了组织张量使其能够以这种方式批处理，它们可能需要用多步 `view`、`reshape` 和 `transpose` 来重塑。这可能有点麻烦，而且常常很难看清代码在做什么以及你的张量的形状是什么。

一个更符合人体工程学的选择是使用 `torch.einsum` 中的 **einsum 记法**，或者更确切地说，使用框架无关的库如 `einops` 或 `einx`。两个关键操作是 `einsum`，它可以对输入张量的任意维度进行张量收缩（tensor contraction），以及 `rearrange`，它可以重排、拼接和分割任意维度。事实证明，几乎所有机器学习中的操作都是维度重排和张量收缩的某种组合，加上偶尔的（通常是逐元素的）非线性函数。这意味着使用 einsum 记法可以让你的很多代码更易读、更灵活。

我们**强烈**建议在本课程中学习并使用 einsum 记法。以前没有接触过 einsum 记法的学生应使用 `einops`（文档见此），而已经熟悉 `einops` 的学生应学习更通用的 `einx`（文档见此）。⁴ 这两个包都已安装在我们提供的环境中。

这里我们给出一些如何使用 einsum 记法的示例。这些是对 einops 文档的补充，你应该先阅读 einops 文档。

> **示例 (einstein_example1)：用 einops.einsum 进行批量矩阵乘法**
>
> ```python
> import torch
> from einops import rearrange, einsum
>
> ## 基本实现
> Y = D @ A.T
> # 难以说明输入和输出的形状以及它们的含义。
> # D 和 A 可以有哪些形状，它们中是否有任何意外行为？
>
> ## Einsum 是自文档化且健壮的
> #                       D                    A         ->            Y
> Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
>
> ## 或者，一个批量版本，其中 D 可以有任意前导维度，但 A 受限。
> Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
> ```

> **示例 (einstein_example2)：用 einops.rearrange 进行广播操作**
>
> 我们有一批图像，对于每张图像我们想根据某个缩放因子生成 10 个变暗版本：
>
> ```python
> images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
> dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
>
> ## 重塑并相乘
> dim_value = rearrange(dim_by,   "dim_value            -> 1 dim_value 1 1 1")
> images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
> dimmed_images = images_rearr * dim_value
>
> ## 或者一步到位：
> dimmed_images = einsum(
>     images, dim_by,
>     "batch height width channel, dim_value -> batch dim_value height width channel"
> )
> ```

> **示例 (einstein_example3)：用 einops.rearrange 进行像素混合**
>
> 假设我们有一批表示为形状 `(batch, height, width, channel)` 张量的图像，我们想对图像的所有像素进行线性变换，但这个变换应对每个通道独立进行。我们的线性变换由一个形状为 `(height * width, height * width)` 的矩阵 $B$ 表示。
>
> ```python
> channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
> B = torch.randn(32*32, 32*32)
>
> ## 重排图像张量以在所有像素间混合
> channels_last_flat = channels_last.view(
>     -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
> )
> channels_first_flat = channels_last_flat.transpose(1, 2)
> channels_first_flat_transformed = channels_first_flat @ B.T
> channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
> channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
> ```
>
> 相反，使用 einops：
>
> ```python
> height = width = 32
> ## Rearrange 取代了笨拙的 torch view + transpose
> channels_first = rearrange(
>     channels_last,
>     "batch height width channel -> batch channel (height width)"
> )
> channels_first_transformed = einsum(
>     channels_first, B,
>     "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
> )
> channels_last_transformed = rearrange(
>     channels_first_transformed,
>     "batch channel (height width) -> batch height width channel",
>     height=height, width=width
> )
> ```
>
> 或者，如果你想疯狂一点：用 `einx.dot`（einx 中等价于 einops.einsum）一步到位：
>
> ```python
> height = width = 32
> channels_last_transformed = einx.dot(
>     "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
>     "-> batch row_out col_out channel",
>     channels_last, B,
>     col_in=width, col_out=width
> )
> ```
>
> 这里的第一个实现可以通过在前后放置注释来说明输入和输出形状而得到改进，但这很笨拙且容易出错。使用 einsum 记法，**文档即实现**！

einsum 记法可以处理任意的输入批处理维度，但它还有一个关键好处，即**自文档化**。使用 einsum 记法的代码中，你的输入和输出张量的相关形状要清晰得多。对于其余的张量，你可以考虑使用张量类型提示，例如使用 `jaxtyping` 库（不特定于 JAX）。

我们将在作业 2 中更多地讨论使用 einsum 记法的性能影响，但现在只需知道它们几乎总是比替代方案更好！

#### 3.2.1 数学记法与内存排布

许多机器学习论文在其记法中使用行向量，这产生的表示与 NumPy 和 PyTorch 默认使用的行主序（row-major）内存排布契合得很好。使用行向量时，线性变换看起来像

$$y = xW^\top, \tag{1}$$

其中行主序 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，行向量 $x \in \mathbb{R}^{1 \times d_{in}}$。注意，这让我们可以通过增加 $x$ 的最外层维度来批处理输入，意味着我们可以用矩阵输入 $X \in \mathbb{R}^{batch \times d_{in}}$ 替换向量输入 $x$。

在线性代数中，更常用的是列向量，其中线性变换看起来像

$$y = Wx, \tag{2}$$

给定行主序 $W \in \mathbb{R}^{d_{out} \times d_{in}}$ 和列向量 $x \in \mathbb{R}^{d_{in}}$。要在这种设定下批处理输入，批处理维度需要放在最后，所以 $x$ 需要被替换为矩阵 $\bar{X} \in \mathbb{R}^{d_{in} \times batch}$。

**在本作业中，我们在数学记法上将主要使用列向量**，因为数学通常遵循这种记法。你应该记住，如果你想使用普通的矩阵乘法，你必须像公式 1 中的行向量约定那样应用转置，因为 PyTorch 使用行主序内存排布。如果你在线性代数操作中使用 `einsum`，只要你正确标注坐标轴，这就不成问题。顺便一提，值得注意的是其他语言/线性代数包如 Matlab、Julia 和 Fortran 都使用列主序内存排布，意味着**批处理维度放在最后**，而 Python 及相关包采用了 C 标准的行主序排布。

### 3.3 基本构建块：Linear 和 Embedding 模块

#### 3.3.1 参数初始化

有效地训练神经网络通常需要仔细初始化模型参数——糟糕的初始化会导致不良行为，如梯度消失或爆炸。Pre-norm transformer 对初始化异常鲁棒，但初始化仍然会对训练速度和收敛产生重大影响。由于本作业已经很长了，我们将把细节留到作业 3，而在此给你一些在大多数情况下都能很好工作的近似初始化。现在，使用：

* Linear 权重：$\mathcal{N}\left(\mu = 0, \sigma^2 = \frac{2}{d_{in}+d_{out}}\right)$，在 $[-3\sigma, 3\sigma]$ 处截断。
* Embedding：$\mathcal{N}(\mu = 0, \sigma^2 = 1)$，在 $[-3, 3]$ 处截断。
* RMSNorm：`1`

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

#### 3.3.2 Linear 模块

线性层是 Transformer 和一般神经网络的基本构建块。首先，你将实现你自己的 `Linear` 类，它继承自 `torch.nn.Module` 并执行线性变换：

$$y = Wx. \tag{3}$$

注意，我们不包含偏置项，遵循大多数现代 LLM。

> **问题 (linear)：实现 Linear 模块（1 分）**
>
> **交付物**：实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，只是没有 `bias` 参数。我们推荐以下接口：
>
> **`def __init__(self, in_features, out_features, device=None, dtype=None)`** 构造一个线性变换模块。此函数应接受以下参数：
>
> * `in_features: int` 输入的最终维度
> * `out_features: int` 输出的最终维度
> * `device: torch.device | None = None` 存储参数的设备
> * `dtype: torch.dtype | None = None` 参数的数据类型
>
> **`def forward(self, x: torch.Tensor) -> torch.Tensor`** 对输入应用线性变换。
>
> 确保：
>
> * 继承 `nn.Module`
> * 调用父类构造函数
> * 将你的参数构造并存储为 $W$（而非 $W^\top$），放入一个 `nn.Parameter`
> * 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`
>
> 对于初始化，使用上面的设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。
>
> 要测试你的 `Linear` 模块，实现测试适配器 **[adapters.run_linear]**。适配器应将给定权重加载到你的 `Linear` 模块中。你可以为此使用 `Module.load_state_dict`。然后，运行 `uv run pytest -k test_linear`。

#### 3.3.3 Embedding 模块

如上所述，Transformer 的第一层是一个嵌入层，它将整数 token ID 映射到维度为 `d_model` 的向量空间。我们将实现一个继承自 `torch.nn.Module` 的自定义 `Embedding` 类（所以你不应使用 `nn.Embedding`）。`forward` 方法应通过使用形状为 `(batch_size, sequence_length)` 的 token ID 的 `torch.LongTensor` 索引到形状为 `(vocab_size, d_model)` 的嵌入矩阵中，来为每个 token ID 选择嵌入向量。

> **问题 (embedding)：实现 Embedding 模块（1 分）**
>
> **交付物**：实现继承自 `torch.nn.Module` 并执行嵌入查找的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：
>
> **`def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)`** 构造一个嵌入模块。此函数应接受以下参数：
>
> * `num_embeddings: int` 词汇表大小
> * `embedding_dim: int` 嵌入向量的维度，即 $d_{model}$
> * `device: torch.device | None = None` 存储参数的设备
> * `dtype: torch.dtype | None = None` 参数的数据类型
>
> **`def forward(self, token_ids: torch.Tensor) -> torch.Tensor`** 查找给定 token ID 的嵌入向量。
>
> 确保：
>
> * 继承 `nn.Module`
> * 调用父类构造函数
> * 将你的嵌入矩阵初始化为 `nn.Parameter`
> * 存储嵌入矩阵时以 `d_model` 作为最终维度
> * 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`
>
> 同样，使用上面的设置进行初始化，并使用 `torch.nn.init.trunc_normal_` 来初始化权重。
>
> 要测试你的实现，实现测试适配器 **[adapters.run_embedding]**。然后，运行 `uv run pytest -k test_embedding`。

### 3.4 Pre-Norm Transformer 块

每个 Transformer 块有两个子层：一个多头自注意力机制和一个逐位置前馈网络（[A. Vaswani et al., 2017]，第 3.1 节）。

在原始 Transformer 论文中，模型在两个子层的每一个周围使用残差连接，然后是层归一化。这种架构通常被称为"post-norm" Transformer，因为层归一化被应用于子层输出。然而，大量工作发现，将层归一化从每个子层的输出移到每个子层的输入（并在最后一个 Transformer 块之后额外加一个层归一化）能提高 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019；R. Xiong et al., 2020]——参见图 2 了解这个"pre-norm" Transformer 块的可视化表示。然后每个 Transformer 块子层的输出通过残差连接被加到子层输入上（[A. Vaswani et al. [8]]，第 5.4 节）。pre-norm 的一个直觉是，从输入嵌入到 Transformer 最终输出之间有一条干净的、没有任何归一化的"残差流"，据称这能改善梯度流。这种 pre-norm Transformer 现在是当今语言模型中使用的标准（如 GPT-3、LLaMA、PaLM 等），所以我们将实现这个变体。我们将逐一介绍 pre-norm Transformer 块的每个组件，并依次实现它们。

#### 3.4.1 均方根层归一化（RMSNorm）

[A. Vaswani et al. [8]] 的原始 Transformer 实现使用层归一化 [J. L. Ba et al., 2016] 来归一化激活。遵循 [H. Touvron et al. [12]]，我们将使用均方根层归一化（RMSNorm；[B. Zhang et al. [13]]，公式 4）进行层归一化。给定一个激活向量 $a \in \mathbb{R}^{d_{model}}$，RMSNorm 将按如下方式重新缩放每个激活 $a_i$：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} g_i, \tag{4}$$

其中 $\text{RMS}(a) = \sqrt{\frac{1}{d_{model}} \sum_{i=1}^{d_{model}} a_i^2 + \varepsilon}$。这里，$g_i$ 是一个可学习的"增益（gain）"参数（共有 `d_model` 个这样的参数），$\varepsilon$ 是一个通常固定为 1e-5 的超参数。

你应该将输入上转（upcast）为 `torch.float32` 以防止在平方输入时溢出。总体上，你的 `forward` 方法应类似于：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# 你在此执行 RMSNorm 的代码
...
result = ...

# 以原始 dtype 返回结果
return result.to(in_dtype)
```

> **问题 (rmsnorm)：均方根层归一化（1 分）**
>
> **交付物**：将 RMSNorm 实现为一个 `torch.nn.Module`。我们推荐以下接口：
>
> **`def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)`** 构造 RMSNorm 模块。此函数应接受以下参数：
>
> * `d_model: int` 模型的隐藏维度
> * `eps: float = 1e-5` 数值稳定性的 epsilon 值
> * `device: torch.device | None = None` 存储参数的设备
> * `dtype: torch.dtype | None = None` 参数的数据类型
>
> **`def forward(self, x: torch.Tensor) -> torch.Tensor`** 处理一个形状为 `(batch_size, sequence_length, d_model)` 的输入张量并返回相同形状的张量。
>
> **注意**：记住在执行归一化之前将输入上转为 `torch.float32`（之后再下转回原始 dtype），如上所述。
>
> 要测试你的实现，实现测试适配器 **[adapters.run_rmsnorm]**。然后，运行 `uv run pytest -k test_rmsnorm`。

#### 3.4.2 逐位置前馈网络

**图 3**：比较 SiLU（又称 Swish）和 ReLU 激活函数。（SiLU: $f(x) = x \cdot \sigma(x)$；Identity: $f(x) = x$；ReLU: $f(x) = \max(0, x)$）

在原始 Transformer 论文（[A. Vaswani et al. [8]] 第 3.3 节）中，Transformer 前馈网络由两个线性变换组成，中间夹一个 ReLU 激活（$\text{ReLU}(x) = \max(0, x)$）。在那个原始架构中，内层前馈层的维度通常是输入维度的 4 倍。

然而，现代语言模型相比这个原始设计倾向于引入两个主要变化：它们使用另一种激活函数并采用门控机制。具体来说，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLM 中采用的"SwiGLU"激活函数，它将 SiLU（通常称为 Swish）激活与一个称为门控线性单元（Gated Linear Unit，GLU）的门控机制结合。我们还将省略线性层中有时使用的偏置项，遵循 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来的大多数现代 LLM。

SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016；S. Elfwing et al., 2017] 定义如下：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}. \tag{5}$$

如图 3 所示，SiLU 激活函数与 ReLU 激活函数相似，但在零点处是平滑的。

门控线性单元（GLU）最初由 [Y. N. Dauphin et al. [19]] 定义，为一个线性变换经过 sigmoid 函数后与另一个线性变换的逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x, \tag{6}$$

其中 $\odot$ 表示逐元素乘法。门控线性单元被认为可以"通过为梯度提供线性路径同时保留非线性能力，来减少深层架构的梯度消失问题"。

将 SiLU/Swish 和 GLU 结合起来，我们得到 SwiGLU，我们将其用于前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1 x) \odot W_3 x), \tag{7}$$

其中 $x \in \mathbb{R}^{d_{model}}$，$W_1, W_3 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$，通常 $d_{ff} = \frac{8}{3} d_{model}$。对于具体实现，将其舍入到 64 的邻近倍数以提高硬件效率是可以的。

[N. Shazeer [20]] 首先提出将 SiLU/Swish 激活与 GLU 结合，并进行了实验表明 SwiGLU 在语言建模任务上优于 ReLU 和 SiLU（无门控）等基线。稍后在作业中，你将比较 SwiGLU 和 SiLU。虽然我们已经提到了这些组件的一些启发式论证（论文也提供了更多支持证据），但保持一个经验视角是好的：来自 Shazeer 论文的一句著名引言是

> "我们不对为什么这些架构似乎有效提供任何解释；我们将其成功，如同一切其他一样，归于神明的恩典。"

> **问题 (positionwise_feedforward)：实现逐位置前馈网络（2 分）**
>
> **交付物**：实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。
>
> **注意**：在这种特定情况下，你可以在实现中使用 `torch.sigmoid` 以获得数值稳定性。
>
> 你应该在实现中将 $d_{ff}$ 设为约 $\frac{8}{3} \times d_{model}$，同时确保内层前馈层的维度是 64 的倍数以充分利用你的硬件。要针对我们提供的测试来测试你的实现，你需要实现测试适配器 **[adapters.run_swiglu]**。然后，运行 `uv run pytest -k test_swiglu` 来测试你的实现。

#### 3.4.3 相对位置嵌入

为了向模型注入位置信息，我们将实现旋转位置嵌入（Rotary Position Embeddings，[J. Su et al., 2021]），通常称为 RoPE。对于位于 token 位置 $i$ 的给定查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们将应用一个成对旋转矩阵 $R^i$，得到 $q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$。这里，$R^i$ 会将嵌入元素对 $q^{(i)}_{2k-1:2k}$ 作为 2 维向量按角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$ 旋转，其中 $k \in \{1, ..., d/2\}$，$\Theta$ 是某个常数。因此，我们可以把 $R^i$ 视为一个大小为 $d \times d$ 的块对角矩阵，其块 $R^i_k$（$k \in \{1, ..., \frac{d}{2}\}$）为

$$R^i_k = \begin{pmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{pmatrix} \tag{8}$$

因此我们得到完整的旋转矩阵

$$R^i = \begin{pmatrix} R^i_1 & 0 & 0 & \cdots & 0 \\ 0 & R^i_2 & 0 & \cdots & 0 \\ 0 & 0 & R^i_3 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & R^i_{d/2} \end{pmatrix}, \tag{9}$$

其中 0 表示 $2 \times 2$ 的零矩阵。虽然你可以构造完整的 $d \times d$ 矩阵，但一个好的解决方案应利用这个矩阵的性质来更高效地实现变换。由于我们只关心一个序列内 token 的相对旋转，我们可以在各层、不同批次之间复用为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算的值。如果你想优化它，可以使用一个由所有层引用的单个 RoPE 模块，它可以有一个用 `self.register_buffer(persistent=False)` 在 init 时创建的 2 维预计算的 sin 和 cos 值缓冲区，而不是 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。我们对 $q^{(i)}$ 所做的完全相同的旋转过程随后也对 $k^{(j)}$ 进行，按对应的 $R^j$ 旋转。注意，这一层没有可学习的参数。

> **问题 (rope)：实现 RoPE（2 分）**
>
> **交付物**：实现一个类 `RotaryPositionalEmbedding`，它对输入张量应用 RoPE。
>
> 推荐以下接口：
>
> **`def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)`** 构造 RoPE 模块并在需要时创建缓冲区。
>
> * `theta: float` RoPE 的 $\Theta$ 值
> * `d_k: int` 查询和键向量的维度
> * `max_seq_len: int` 将要输入的最大序列长度
> * `device: torch.device | None = None` 存储缓冲区的设备
>
> **`def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`** 处理一个形状为 `(..., seq_len, d_k)` 的输入张量并返回相同形状的张量。注意你应该容忍 $x$ 有任意数量的批处理维度。你应该假设 token 位置是一个形状为 `(..., seq_len)` 的张量，它指定 $x$ 沿序列维度的 token 位置。
>
> 你应该使用 token 位置来沿序列维度切片你的（可能预计算的）cos 和 sin 张量。
>
> 要测试你的实现，完成 **[adapters.run_rope]** 并确保它通过 `uv run pytest -k test_rope`。

#### 3.4.4 缩放点积注意力

我们现在将实现 [A. Vaswani et al. [8]]（第 3.2.1 节）中描述的缩放点积注意力。作为初步步骤，Attention 操作的定义将用到 softmax，这是一个把未归一化的分数向量转换为归一化分布的操作：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}. \tag{10}$$

注意，对于较大的值，$\exp(v_i)$ 可能变成 `inf`（此时 $\frac{\text{inf}}{\text{inf}} = \text{NaN}$）。我们可以通过注意到 softmax 操作对给所有输入加上任意常数 $c$ 是不变的来避免这一点。我们可以利用这个性质来获得数值稳定性——通常，我们会从 $v$ 的所有元素中减去 $v$ 的最大条目，使新的最大条目为 0。你现在将实现 softmax，使用这个技巧来获得数值稳定性。

> **问题 (softmax)：实现 softmax（1 分）**
>
> **交付物**：编写一个函数在张量上应用 softmax 操作。你的函数应接受两个参数：一个张量和一个**维度 $i$**，并对输入张量的第 $i$ 个维度应用 softmax。输出张量应与输入张量形状相同，但其第 $i$ 个维度现在将有一个归一化的概率分布。使用从第 $i$ 个维度的所有元素中减去最大值的技巧来避免数值稳定性问题。
>
> 要测试你的实现，完成 **[adapters.run_softmax]** 并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

现在我们可以在数学上定义 Attention 操作如下：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{11}$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里，$Q$、$K$ 和 $V$ 都是此操作的输入——注意它们不是可学习的参数。

**掩码（Masking）**：有时对注意力操作的输出进行**掩码**是方便的。掩码应有形状 $M \in \{\text{True}, \text{False}\}^{n \times m}$，这个布尔矩阵的每一行 $i$ 指示查询 $i$ 应关注哪些键。规范地（也有点令人困惑地），位置 $(i, j)$ 处的值 `True` 表示查询 $i$ **确实**关注键 $j$，值 `False` 表示查询 **不**关注键。换句话说，信息在值为 `True` 的 $(i, j)$ 对处"流动"。例如，考虑一个条目为 $[[\text{True}, \text{True}, \text{False}]]$ 的 $1 \times 3$ 掩码矩阵。这个单查询向量只关注前两个键。

在计算上，使用掩码会比在子序列上计算注意力高效得多，我们可以通过取 softmax 前的值 $\left(QK^\top / \sqrt{d_k}\right)$ 并给掩码矩阵中任何为 `False` 的条目加上 $-\infty$ 来做到这一点。

> **问题 (scaled_dot_product_attention)：实现缩放点积注意力（5 分）**
>
> **交付物**：实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 表示任意数量的其他批处理式维度（如果提供）。实现应返回形状为 `(batch_size, ..., seq_len, d_v)` 的输出。关于批处理式维度的讨论，见第 3.2 节。
>
> 你的实现还应支持一个可选的用户提供的形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 `True` 的位置的注意力概率应共同求和为 1，掩码值为 `False` 的位置的注意力概率应为零。
>
> 要针对我们提供的测试来测试你的实现，你需要实现测试适配器 **[adapters.run_scaled_dot_product_attention]**。`uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试你的实现。

#### 3.4.5 因果多头自注意力

我们将实现 [A. Vaswani et al. [8]] 第 3.2.2 节中描述的多头自注意力。回忆一下，从数学上讲，应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) \tag{12}$$

$$\text{for head}_i = \text{Attention}(Q_i, K_i, V_i) \tag{13}$$

其中 $Q_i$、$K_i$、$V_i$ 分别是 $Q$、$K$、$V$ 的嵌入维度中大小为 $d_k$ 或 $d_v$ 的第 $i$ 个切片（$i \in \{1, ..., h\}$）。Attention 为第 3.4.4 节中定义的缩放点积注意力操作。由此我们可以形成多头**自**注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHead}(W_Q x, W_K x, W_V x) \tag{14}$$

这里，可学习的参数是 $W_Q \in \mathbb{R}^{hd_k \times d_{model}}$、$W_K \in \mathbb{R}^{hd_k \times d_{model}}$、$W_V \in \mathbb{R}^{hd_v \times d_{model}}$ 和 $W_O \in \mathbb{R}^{d_{model} \times hd_v}$。由于在多头注意力操作中 $Q$、$K$ 和 $V$ 被切片，我们可以把 $W_Q$、$W_K$ 和 $W_V$ 视为沿输出维度为每个头分离。当你完成这个工作时，你应该在总共三次矩阵乘法中计算键、值和查询投影。⁵

**因果掩码**

你的实现应防止模型关注序列中的未来 token。换句话说，如果模型被给予一个 token 序列 $t_1, ..., t_n$，我们想计算前缀 $t_1, ..., t_i$（其中 $i < n$）的下一个词预测，模型应**不能**访问（关注）位置 $t_{i+1}, ..., t_n$ 的 token 表示，因为在推理期间生成文本时它将无法访问这些 token（并且这些未来 token 会泄露关于真实下一个词身份的信息，使语言建模预训练目标变得平凡）。对于输入 token 序列 $t_1, ..., t_n$，我们可以通过运行多头自注意力 $n$ 次（对 $n$ 个唯一前缀）来朴素地防止对未来 token 的访问。相反，我们将使用因果注意力掩码，它允许 token $i$ 关注序列中所有位置 $j \le i$。你可以使用 `torch.triu` 或广播索引比较来构造这个掩码，并且你应该利用你在第 3.4.4 节的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**

RoPE 应应用于查询和键向量，但不应用于值向量。此外，头维度应作为批处理维度处理，因为在多头注意力中，注意力是对每个头独立应用的。这意味着应对每个头的查询和键向量应用完全相同的 RoPE 旋转。

> **问题 (multihead_self_attention)：实现因果多头自注意力（5 分）**
>
> **交付物**：将因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应（至少）接受以下参数：
>
> * `d_model: int` Transformer 块输入的维度。
> * `num_heads: int` 多头自注意力中使用的头数。
>
> 遵循 [A. Vaswani et al. [8]]，设 $d_k = d_v = \frac{d_{model}}{h}$。要针对我们提供的测试来测试你的实现，实现测试适配器 **[adapters.run_multihead_self_attention]**。然后，运行 `uv run pytest -k test_multihead_self_attention` 来测试你的实现。

### 3.5 完整的 Transformer LM

让我们从组装 Transformer 块开始（参考图 2 会有帮助）。一个 Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于 SwiGLU 前馈网络。在每个子层中，我们首先执行 RMSNorm，然后是主操作（MHA/FF），最后加上残差连接。

具体地说，Transformer 块的前半部分（第一个"子层"）应实现以下更新集，以从输入 $x$ 产生输出 $y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)). \tag{15}$$

> **问题 (transformer_block)：实现 Transformer 块（3 分）**
>
> 实现第 3.4 节中描述并在图 2 中所示的 pre-norm Transformer 块。你的 Transformer 块应（至少）接受以下参数。
>
> * `d_model: int` Transformer 块输入的维度。
> * `num_heads: int` 多头自注意力中使用的头数。
> * `d_ff: int` 逐位置前馈内层的维度。
>
> 要测试你的实现，实现适配器 **[adapters.run_transformer_block]**。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。
>
> **交付物**：通过所提供测试的 Transformer 块代码。

现在我们把这些块组合起来，遵循图 1 中的高层图示。按照第 3.1.0.1 节中对嵌入的描述，将其输入到 `num_layers` 个 Transformer 块中，然后将其传入最后的层归一化和 LM 头，以获得词汇表上的一个未归一化分布（logits）。

> **问题 (transformer_lm)：实现 Transformer LM（3 分）**
>
> 是时候把一切组合起来了！实现第 3.1 节中描述并在图 1 中所示的 Transformer 语言模型。至少，你的实现应接受上述所有 Transformer 块的构造参数，以及这些额外参数：
>
> * `vocab_size: int` 词汇表的大小，用于确定 token 嵌入矩阵的维度。
> * `context_length: int` 最大上下文长度，用于确定 RoPE sin 和 cos 缓冲区的维度。
> * `num_layers: int` 要使用的 Transformer 块的数量。
>
> 要针对我们提供的测试来测试你的实现，你首先需要实现测试适配器 **[adapters.run_transformer_lm]**。然后，运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。
>
> **交付物**：通过上述测试的 Transformer LM 模块。

**资源核算**

能够理解 Transformer 的各个部分如何消耗计算和内存是很有用的。我们将通过一些基本的"FLOPs 核算"步骤。Transformer 中**绝大多数**的 FLOPS 是矩阵乘法，所以我们的核心方法很简单：

1. 写下 Transformer 前向传递中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实会很有用：

> **规则**：给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵-矩阵乘积 $AB$ 需要 $2mnp$ 个 FLOPs。

要看出这一点，注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，而这个点积需要 $n$ 次加法和 $n$ 次乘法（$2n$ FLOPs）。然后，由于矩阵-矩阵乘积 $AB$ 有 $m \times p$ 个条目，FLOPS 总数为 $(2n)(mp) = 2mnp$。

现在，在你做下一个问题之前，逐个检查你的 Transformer 块和 Transformer LM 的每个组件，并列出所有矩阵乘法及其相关的 FLOPs 成本，可能会有帮助。

> **问题 (transformer_accounting)：Transformer LM 资源核算（5 分）**
>
> **(a)** 考虑一个使用我们作业架构的 GPT-2 XL 大小的模型，其配置如下：
>
> * `vocab_size`: 50,257
> * `context_length`: 1,024
> * `num_layers`: 48
> * `d_model`: 1,600
> * `num_heads`: 25
> * `d_ff`: 4,288（$\frac{8}{3} \times 1,600$ 的邻近 64 倍数）
>
> 假设我们使用这个配置构造模型。我们的模型将有多少可训练参数？假设每个参数用单精度浮点表示，仅加载此模型需要多少内存？
>
> **交付物**：一到两句话回答。
>
> **(b)** 识别完成我们 GPT-2 XL 形状模型的一次前向传递所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列有 `context_length` 个 token。
>
> **交付物**：矩阵乘法列表（附描述），以及所需的 FLOPs 总数。
>
> **(c)** 基于你上面的分析，模型的哪些部分需要最多的 FLOPs？
>
> **交付物**：一到两句话回答。
>
> **(d)** 对 GPT-2 small（12 层，768 `d_model`，12 头）、GPT-2 medium（24 层，1024 `d_model`，16 头）和 GPT-2 large（36 层，1280 `d_model`，20 头）重复你的分析。随着模型规模增加，Transformer LM 的哪些部分在总 FLOPs 中占比成比例地更多或更少？
>
> **交付物**：对于每个模型，提供模型组件及其相关 FLOPs 的分解（作为前向传递所需总 FLOPs 的比例）。此外，用一到两句话描述改变模型规模如何改变每个组件的比例 FLOPs。
>
> **(e)** 取 GPT-2 XL 并将上下文长度增加到 16,384。一次前向传递的总 FLOPs 如何变化？模型组件的相对 FLOPs 贡献如何变化？
>
> **交付物**：一到两句话回答。

> ⁴ 值得注意的是，虽然 `einops` 有大量的支持，但 `einx` 没有那么经过实战检验。如果你发现 `einx` 中有任何限制或 bug，请随时回退到使用 `einops` 加上一些更普通的 PyTorch。
>
> ⁵ 作为一个进阶目标，尝试将键、查询和值投影组合成单个权重矩阵，这样你只需要一次矩阵乘法。

---

## 4 训练 Transformer LM

现在我们有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的是构建支持训练的所有代码。这包括以下内容：

* **损失**：我们需要定义损失函数（交叉熵）。
* **优化器**：我们需要定义优化器来最小化这个损失（AdamW）。
* **训练循环**：我们需要所有加载数据、保存检查点和管理训练的支持基础设施。

### 4.1 交叉熵损失

回忆一下，Transformer 语言模型为每个长度为 $m + 1$ 的序列 $x$ 和 $i = 1, ..., m$ 定义了一个分布 $p_\theta(x_{i+1} \mid x_{1:i})$。给定一个由长度为 $m + 1$ 的序列组成的训练集 $D$，我们定义标准的交叉熵（负对数似然）损失函数：

$$\ell(\theta; D) = \frac{1}{|D| m} \sum_{x \in D} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} \mid x_{1:i}). \tag{16}$$

（注意，Transformer 中一次前向传递对**所有** $i = 1, ..., m$ 产生 $p_\theta(x_{i+1} \mid x_{1:i})$。）

特别地，Transformer 为每个位置 $i$ 计算 logits $o_i \in \mathbb{R}^{\text{vocab\_size}}$，这产生：⁶

$$p(x_{i+1} \mid x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}. \tag{17}$$

交叉熵损失通常相对于 logits 向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$ 和目标 $x_{i+1}$ 来定义。⁷

实现交叉熵损失需要对数值问题格外小心，就像在 softmax 的情况下一样。

> **问题 (cross_entropy)：实现交叉熵（1 分）**
>
> **交付物**：编写一个函数计算交叉熵损失，它接受预测的 logits（$o_i$）和目标（$x_{i+1}$）并计算交叉熵 $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下情况：
>
> * 为数值稳定性减去最大元素。
> * 尽可能抵消 log 和 exp。
> * 处理任何额外的批处理维度并返回批次上的**平均值**。与第 3.2 节一样，我们假设批处理式维度总是在最前面，在词汇表大小维度之前。
>
> 实现 **[adapters.run_cross_entropy]**，然后运行 `uv run pytest -k test_cross_entropy` 来测试你的实现。

**困惑度（Perplexity）**

交叉熵足以用于训练，但当我们评估模型时，我们还想报告困惑度。对于一个长度为 $m$ 的序列，我们承受交叉熵损失 $\ell_1, ..., \ell_m$：

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right). \tag{18}$$

### 4.2 SGD 优化器

现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t = 0, ..., T - 1$，我们执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \tag{19}$$

其中 $B_t$ 是从数据集 $D$ 采样的一个随机数据批次，**学习率** $\alpha_t$ 和**批次大小** $|B_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

要实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。一个 `Optimizer` 子类必须实现两个方法：

**`def __init__(self, params, ...)`** 应初始化你的优化器。这里，`params` 将是要优化的参数集合（或参数组，以防用户想对模型的不同部分使用不同的超参数，如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，它将存储这些参数以供 `step` 使用。你可以根据优化器接受额外参数（如学习率是一个常见的），并将它们作为字典传递给基类构造函数，字典的键是你为这些参数选择的名称（字符串）。

**`def step(self)`** 应对参数进行一次更新。在训练循环中，这将在反向传递之后被调用，所以你可以访问最后一个批次的梯度。此方法应遍历每个参数张量 $p$ 并**就地**修改它们，即设置 `p.data`，它持有基于梯度 `p.grad`（如果存在）与该参数相关的张量，`p.grad` 是损失相对于该参数的梯度张量。

PyTorch 优化器 API 有几个微妙之处，所以用一个例子来解释会更容易。为了使我们的例子更丰富，我们将实现一个 SGD 的略微变体，其中学习率随训练衰减，从初始学习率 $\alpha$ 开始，随时间取越来越小的步长：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t + 1}} \nabla L(\theta_t; B_t) \tag{20}$$

让我们看看这个版本的 SGD 如何作为 PyTorch `Optimizer` 实现：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # 获取与 p 相关的状态。
                t = state.get("t", 0)  # 从状态获取迭代次数，或 0。
                grad = p.grad.data  # 获取损失相对于 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad  # 就地更新权重张量。
                state["t"] = t + 1  # 递增迭代次数。

        return loss
```

在 `__init__` 中，我们将参数以及默认超参数传递给基类构造函数（参数可以以组的形式传递，每组有不同的超参数）。如果参数只是单个 `torch.nn.Parameter` 对象的集合，基类构造函数将创建一个单一组并为其分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用公式 20。这里，我们保持迭代次数作为与每个参数相关的状态：我们首先读取这个值，在梯度更新中使用它，然后更新它。API 指定用户可能传入一个可调用的 `closure` 来在优化器步骤之前重新计算损失。我们不会为我们要使用的优化器用到这个，但我们添加它以遵循 API。

要看到这个工作，我们可以使用以下**训练循环**的最小示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad()  # 为所有可学习参数重置梯度。
    loss = (weights**2).mean()  # 计算一个标量损失值。
    print(loss.cpu().item())
    loss.backward()  # 运行反向传递，计算梯度。
    opt.step()  # 运行优化器步骤。
```

这是训练循环的典型结构：在每次迭代中，我们计算损失并运行一步优化器。训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 给我们这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构将是相同的。

> **问题 (learning_rate_tuning)：调整学习率（1 分）**
>
> 正如我们将看到的，影响训练最大的超参数之一是学习率。让我们在我们的玩具示例中实际看看这一点。用三个其他学习率值运行上面的 SGD 示例：1e1、1e2 和 1e3，仅运行 10 次训练迭代。对于每个学习率，损失会发生什么？它是衰减得更快、更慢，还是发散（即在训练过程中增加）？
>
> **交付物**：一到两句话回答，说明你观察到的行为。

### 4.3 AdamW

现代语言模型通常用更复杂的优化器来训练，而不是 SGD。最近使用的大多数优化器是 Adam 优化器 [D. P. Kingma et al., 2015] 的衍生物。我们将使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛使用。AdamW 提出对 Adam 的一个修改，通过添加**权重衰减（weight decay）**来改善正则化（在每次迭代中，我们将参数拉向 0），其方式与梯度更新解耦。我们将实现 [I. Loshchilov et al. [23]] 的算法 2 中描述的 AdamW。

AdamW 是**有状态的**：对于每个参数，它保持其第一和第二矩的运行估计。因此，AdamW 用额外的内存换取改善的稳定性和收敛。除了学习率 $\alpha$，AdamW 有一对控制矩估计更新的超参数 $(\beta_1, \beta_2)$，以及一个权重衰减率 $\lambda$。典型应用将 $(\beta_1, \beta_2)$ 设为 $(0.9, 0.999)$，但像 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 这样的大型语言模型通常用 $(0.9, 0.95)$ 训练。算法可以写成如下，其中 $\varepsilon$ 是一个小值（如 $10^{-8}$），用于在我们得到极小的 $v$ 值时改善数值稳定性：

> **算法 1：AdamW 优化器**
>
> ```
> init(θ)                                   ▷ 初始化可学习参数
> m ← 0                                     ▷ 第一矩向量的初始值；与 θ 形状相同
> v ← 0                                     ▷ 第二矩向量的初始值；与 θ 形状相同
> for t = 1, ..., T do
>     Sample batch of data B_t
>     g ← ∇_θ ℓ(θ; B_t)                     ▷ 计算损失的梯度
>     α_t ← α · √(1 - β₂ᵗ) / (1 - β₁ᵗ)       ▷ 计算迭代 t 的调整后 α
>     θ ← θ - α · λ · θ                      ▷ 应用权重衰减
>     m ← β₁·m + (1 - β₁)·g                  ▷ 更新第一矩估计
>     v ← β₂·v + (1 - β₂)·g²                 ▷ 更新第二矩估计
>     θ ← θ - α_t · m / (√v + ε)             ▷ 应用矩调整的权重更新
> end for
> ```
>
> **算法 1**：AdamW 优化器伪代码。

注意 $t$ 从 1 开始。你现在将实现这个优化器。

> **问题 (adamw)：实现 AdamW（2 分）**
>
> **交付物**：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接受学习率 $\alpha$，以及 $\beta$、$\varepsilon$ 和 $\lambda$ 超参数。为了帮助你保持状态，基类 `Optimizer` 给你一个字典 `self.state`，它将 `nn.Parameter` 对象映射到一个字典，该字典存储你为该参数需要的任何信息（对 AdamW 来说，这将是矩估计）。实现 **[adapters.get_adamw_cls]** 并确保它通过 `uv run pytest -k test_adamw`。

> **问题 (adamw_accounting)：使用 AdamW 训练的资源核算（2 分）**
>
> 让我们计算运行 AdamW 需要多少内存和计算。假设我们对每个张量都使用 float32。
>
> **(a)** 运行 AdamW 需要多少峰值内存？基于参数、激活、梯度和优化器状态的内存使用来分解你的答案。用批次大小和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）来表达你的答案。假设 $d_{ff} = \frac{8}{3} \times d_{model}$。
>
> 为简单起见，在计算激活的内存使用时，只考虑以下组件：
>
> * Transformer 块
>   * RMSNorm(s)
>   * 多头自注意力子层：$QKV$ 投影、$QK^\top$ 矩阵乘法、softmax、值的加权和、输出投影。
>   * 逐位置前馈（SwiGLU）：$W_1$、$W_2$、门控分支上的 SiLU、逐元素乘积、$W_3$
> * 最后的 RMSNorm
> * 输出嵌入
> * logits 上的交叉熵
>
> **交付物**：参数、激活、梯度和优化器状态各自的代数表达式，以及总和。
>
> **(b)** 为 GPT-2 XL 形状的模型实例化你的答案，得到一个仅依赖于 `batch_size` 的表达式。在 80GB 内存内你能使用的最大批次大小是多少？
>
> **交付物**：一个形如 $a \cdot \text{batch\_size} + b$ 的表达式（$a, b$ 为数值），以及一个表示最大批次大小的数字。
>
> **(c)** 运行一步 AdamW 需要多少 FLOPs？
>
> **交付物**：一个代数表达式，附简要说明。
>
> **(d)** 模型 FLOPs 利用率（MFU）定义为观测吞吐量（每秒 token 数）相对于硬件理论峰值 FLOP 吞吐量的比率 [A. Chowdhery et al., 2022]。一块 NVIDIA H100 GPU 对 "float32"（实际上是 TensorFloat-32，实际是 "bfloat19"）有 495 teraFLOP/s 的理论峰值。假设你能达到 50% MFU，在单块 H100 上训练一个 GPT-2 XL 400K 步、批次大小 1024 需要多长时间？遵循 [J. Kaplan et al. [25]] 和 [J. Hoffmann et al. [26]]，假设反向传递的 FLOPs 是前向传递的两倍。
>
> **交付物**：训练需要的小时数，附简要说明。

### 4.4 学习率调度

导致损失最快下降的学习率值通常在训练过程中变化。在训练 Transformer 时，通常使用学习率**调度（schedule）**，我们从较大的学习率开始，在开始时做更快的更新，随着模型训练缓慢衰减到较小的值。⁸ 在本作业中，我们将实现用于训练 LLaMA [H. Touvron et al., 2023] 的余弦退火调度。

调度器是一个函数，它接受当前步 $t$ 和其他相关参数（如初始和最终学习率），并返回在步 $t$ 用于梯度更新的学习率。最简单的调度是常数函数，它对任何 $t$ 都返回相同的学习率。

余弦退火学习率调度接受 (i) 当前迭代 $t$、(ii) 最大学习率 $\alpha_{max}$、(iii) 最小（最终）学习率 $\alpha_{min}$、(iv) **预热（warm-up）**迭代次数 $T_w$，以及 (v) 余弦退火的最终迭代 $T_c$。迭代 $t$ 处的学习率定义为：

**（预热）** 如果 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{max}$。

**（余弦退火）** 如果 $T_w \le t \le T_c$，则 $\alpha_t = \alpha_{min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{max} - \alpha_{min})$。

**（退火后）** 如果 $t > T_c$，则 $\alpha_t = \alpha_{min}$。

> **问题 (learning_rate_schedule)：实现带预热的余弦学习率调度（1 分）**
>
> 编写一个函数，它接受 $t$、$\alpha_{max}$、$\alpha_{min}$、$T_w$ 和 $T_c$，并根据上面定义的调度器返回学习率 $\alpha_t$。然后实现 **[adapters.get_lr_cosine_schedule]** 并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪

在训练期间，我们有时会遇到产生大梯度的训练样本，这可能破坏训练的稳定性。为了缓解这一点，实践中常用的一种技术是**梯度裁剪（gradient clipping）**。其思想是在每次反向传递之后、进行优化器步骤之前，对梯度的范数施加一个限制。

给定梯度（对所有参数）$g$，我们计算其 $\ell_2$-范数 $\|g\|_2$。如果这个范数小于最大值 $M$，那么我们让 $g$ 保持原样；否则，我们将 $g$ 缩小一个因子 $\frac{M}{\|g\|_2 + \varepsilon}$（其中加入一个小的 $\varepsilon$，如 $10^{-6}$，以保证数值稳定性）。注意，结果范数将刚好在 $M$ 以下。

> **问题 (gradient_clipping)：实现梯度裁剪（1 分）**
>
> 编写一个实现梯度裁剪的函数。你的函数应接受一个参数列表和一个最大 $\ell_2$-范数。它应就地修改每个参数梯度。使用 $\varepsilon = 10^{-6}$（PyTorch 默认值）。然后，实现适配器 **[adapters.run_gradient_clipping]** 并确保它通过 `uv run pytest -k test_gradient_clipping`。

> ⁶ 注意 $o_i[k]$ 指向量 $o_i$ 中索引 $k$ 处的值。
>
> ⁷ 这对应于 $x_{i+1}$ 上的狄拉克 delta 分布与预测的 $\text{softmax}(o_i)$ 分布之间的交叉熵。
>
> ⁸ 有时也常用一种学习率会重新升高（重启）的调度，以帮助越过局部最小值。

---

## 5 训练循环

现在我们终于要把目前构建的主要组件组合起来：分词后的数据、模型和优化器。

### 5.1 数据加载器

分词后的数据（例如你在 `tokenizer_experiments` 中准备的）是单个 token 序列 $x = (x_1, ..., x_n)$。尽管源数据可能由单独的文档组成（如不同的网页或源代码文件），一个常见的做法是将所有这些拼接为单个 token 序列，在它们之间添加一个分隔符（如 `<|endoftext|>` token）。

**数据加载器**将其转换为**批次**流，每个批次由 $B$ 个长度为 $m$ 的序列组成，配对相应的下一个 token（长度也为 $m$）。例如，对于 $B = 1$，$m = 3$，$([x_2, x_3, x_4], [x_3, x_4, x_5])$ 会是一个潜在的批次。

以这种方式加载数据出于多个原因简化了训练。首先，任何 $1 \le i \le n - m$ 都给出一个有效的训练序列，所以采样训练序列是平凡的。由于所有训练序列有相同长度，无需填充输入序列，这改善了硬件利用率（也通过增加批次大小 $B$）。最后，我们也无需加载完整数据集来采样训练数据，使处理可能无法放入内存的大型数据集变得容易。

> **问题 (data_loading)：实现数据加载（2 分）**
>
> **交付物**：编写一个函数，它接受一个 numpy 数组 $x$（带 token ID 的整数数组）、一个 `batch_size`、一个 `context_length` 和一个 PyTorch 设备字符串（如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样的输入序列和相应的下一个 token 目标。两个张量都应有形状 `(batch_size, context_length)`，包含 token ID，并且都应放置在请求的设备上。要针对我们提供的测试来测试你的实现，你首先需要实现测试适配器 **[adapters.run_get_batch]**。然后，运行 `uv run pytest -k test_get_batch` 来测试你的实现。

> **低资源提示：在 CPU 或 Apple Silicon 上加载数据**
>
> 如果你计划在 CPU 或 Apple Silicon 上训练你的 LM，你需要把数据移到正确的设备（同样，你稍后也应对模型使用相同的设备）。
>
> 如果你在 CPU 上，可以使用 `'cpu'` 设备字符串，在 Apple Silicon（M* 芯片）上，可以使用 `'mps'` 设备字符串。
>
> 关于 MPS 的更多信息，查看这些资源：
> * `https://docs.pytorch.org/docs/stable/mps.html`
> * `https://docs.pytorch.org/docs/stable/notes/mps.html`
> * `https://developer.apple.com/documentation/metalperformanceshaders`

如果数据集太大无法加载到内存怎么办？我们可以使用一个名为 `mmap` 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存，并在访问该内存位置时惰性加载文件内容。因此，你可以"假装"你把整个数据集放在内存中。Numpy 通过 `np.memmap`（或对 `np.load` 使用 `mmap_mode='r'` 标志，如果你最初用 `np.save` 保存数组）实现这一点，它将返回一个类 numpy 数组对象，在你访问时按需加载条目。**在训练期间从你的数据集（即一个 numpy 数组）采样时，务必以内存映射模式加载数据集**（通过 `np.memmap` 或对 `np.load` 使用 `mmap_mode='r'` 标志，取决于你如何保存数组）。确保你还指定一个与你正在加载的数组匹配的 `dtype`。显式验证内存映射数据看起来正确（例如，不包含超出预期词汇表大小的值）可能会有帮助。

### 5.2 检查点（Checkpointing）

除了加载数据，我们在训练时还需要保存模型。运行作业时，我们常常想要能够恢复一个中途停止的训练运行（例如由于作业超时、机器故障等）。即使一切顺利，我们也可能想稍后访问中间模型（例如，为了事后研究训练动态、从不同训练阶段的模型采样等）。

一个检查点应有我们恢复训练所需的所有状态。我们当然想至少能够恢复模型权重。如果使用有状态的优化器（如 AdamW），我们还需要保存优化器的状态（例如，对 AdamW 来说是矩估计）。最后，为了恢复学习率调度，我们需要知道我们停止时的迭代次数。PyTorch 让保存所有这些变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，返回一个包含所有可学习权重的字典；我们可以稍后用姊妹方法 `load_state_dict()` 恢复这些权重。任何 `torch.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以将一个对象（如包含张量作为某些值、但也有像整数这样的常规 Python 对象的字典）转储到一个文件（路径）或类文件对象，之后可以用 `torch.load(src)` 加载回内存。

> **问题 (checkpointing)：实现模型检查点（1 分）**
>
> 实现以下两个函数来加载和保存检查点：
>
> **`def save_checkpoint(model, optimizer, iteration, out)`** 应将模型、优化器和迭代的所有状态转储到类文件对象 `out`。你可以使用模型和优化器的 `state_dict` 方法来获取它们的相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out`（PyTorch 支持路径或类文件对象）。一个典型选择是让 `obj` 为一个字典，但只要你之后能加载检查点，你可以使用任何你想要的格式。
>
> 此函数期望以下参数：
>
> * `model: torch.nn.Module`
> * `optimizer: torch.optim.Optimizer`
> * `iteration: int`
> * `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
>
> **`def load_checkpoint(src, model, optimizer)`** 应从 `src`（路径或类文件对象）加载一个检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点的迭代次数。你可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，并在模型和优化器中使用 `load_state_dict` 方法将它们恢复到之前的状态。
>
> 此函数期望以下参数：
>
> * `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
> * `model: torch.nn.Module`
> * `optimizer: torch.optim.Optimizer`
>
> 实现 **[adapters.run_save_checkpoint]** 和 **[adapters.run_load_checkpoint]** 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环

现在，终于到了把你实现的所有组件组合到主训练脚本中的时候了。让启动不同超参数的训练运行变得容易会很有回报（例如，将它们作为命令行参数），因为你稍后会做很多次这样的运行来研究不同选择如何影响训练。

> **问题 (training_together)：把它组合起来（4 分）**
>
> **交付物**：编写一个脚本，运行一个训练循环，在用户提供的输入上训练你的模型。特别地，我们建议你的训练脚本允许（至少）以下内容：
>
> * 能够配置和控制各种模型和优化器超参数。
> * 使用 `np.memmap` 内存高效地加载大型训练集和验证集。
> * 将检查点序列化到用户提供的路径。
> * 定期记录训练和验证性能（例如，到控制台和/或像 Weights and Biases 这样的外部服务）。⁹

> ⁹ `wandb.ai`

---

## 6 生成文本

现在我们可以训练模型了，我们需要的最后一块是从模型生成文本的能力。回忆一下，语言模型接受一个（可能批量的）长度为 `sequence_length` 的整数序列，并产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，其中序列的每个元素是一个预测该位置之后下一个 token 的概率分布。我们现在将编写几个函数，把这转换为对新序列的采样方案。

**Softmax**

按照标准约定，语言模型输出是最后一个线性层的输出（"logits"），所以我们必须通过 **softmax** 操作把它转换为归一化概率，这我们之前在公式 10 中见过。

**解码（Decoding）**

要从我们的模型生成文本（解码），我们将向模型提供一个前缀 token 序列（"prompt"），并要求它产生一个在词汇表上预测序列中下一个 token 的概率分布。然后，我们将从这个在词汇表项上的分布中采样，以确定下一个输出 token。

具体地，解码过程的一步应接受一个序列 $x_{1...t}$ 并通过以下等式返回一个 token $x_{t+1}$：

$$P(x_{t+1} = i \mid x_{1...t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)} \tag{21}$$

$$v = \text{TransformerLM}(x_{1...t})_t \in \mathbb{R}^{\text{vocab\_size}} \tag{22}$$

其中 TransformerLM 是我们的模型，它接受一个长度为 `sequence_length` 的序列作为输入并产生一个大小为 `(sequence_length, vocab_size)` 的矩阵，我们取这个矩阵的最后一个元素，因为我们要的是 $t$ 位置的下一个 token 预测。

这给了我们一个基本的解码器，通过反复从这些一步条件分布采样（将我们之前生成的输出 token 追加到下一个解码时间步的输入），直到我们生成序列结束 token `<|endoftext|>`（或用户指定的最大生成 token 数）。

**解码器技巧**

我们将用小模型做实验，而小模型有时会生成质量很低的文本。两个简单的解码器技巧可以帮助修复这些问题。首先，在**温度缩放（temperature scaling）**中，我们用一个温度参数 $\tau$ 修改我们的 softmax，新的 softmax 为：

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{\text{vocab\_size}} \exp(v_j / \tau)}. \tag{23}$$

注意，设置 $\tau \to 0$ 会使 $v$ 的最大元素占主导，softmax 的输出变成一个集中在这个最大元素上的独热向量。

其次，另一个技巧是**核（nucleus）**或 **top-p** 采样，我们通过截断低概率 token 来修改采样分布。设 $q$ 为我们从（温度缩放的）softmax 得到的一个大小为 `vocab_size` 的概率分布。带超参数 $p$ 的核采样根据以下等式产生下一个 token：

$$P(x_{t+1} = i \mid q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{如果 } i \in V(p) \\ 0 & \text{否则} \end{cases} \tag{24}$$

其中 $V(p)$ 是满足 $\sum_{j \in V(p)} q_j \ge p$ 的**最小**索引集。你可以通过首先按大小排序概率分布 $q$，选择最大的词汇表元素直到达到目标水平 $p$，来轻松计算这个量。

> **问题 (decoding)：解码（3 分）**
>
> **交付物**：实现一个从你的语言模型解码的函数。我们建议你支持以下功能：
>
> * 为用户提供的 prompt 生成补全（即接受某个 $x_{1..t}$ 并采样一个补全，直到你遇到 `<|endoftext|>` token）。
> * 允许用户控制生成 token 的最大数量。
> * 给定一个期望的温度值，在采样前对预测的下一个 token 分布应用 softmax 温度缩放。
> * Top-$p$ 采样（[A. Holtzman et al., 2020]，也称为核采样），给定一个用户指定的阈值。

---

## 7 实验

现在是时候把一切组合起来，在预训练数据集上训练（小型）语言模型了。

### 7.1 如何运行实验和交付物

理解 Transformer 架构组件背后原理的最好方法是实际修改它并自己运行。没有什么能替代动手经验。

为此，能够**快速、一致地实验并保存记录**你所做的事很重要。为了快速实验，我们将在一个小规模模型（约 17M 总参数）和简单数据集（TinyStories）上运行许多实验。为了一致地做事，你将系统地消融组件并改变超参数，为了保存记录，我们会要求你提交一份你的实验日志和每个实验相关的学习曲线。

为了能够提交损失曲线，**确保定期评估验证损失并记录步数和挂钟时间（wall-clock times）**。你可能会发现像 Weights and Biases 这样的日志基础设施有帮助。

> **问题 (experiment_log)：实验日志（3 分）**
>
> 为你的训练和评估代码创建实验跟踪基础设施，允许你相对于梯度步数和挂钟时间跟踪你的实验和损失曲线。
>
> **交付物**：你的实验的日志基础设施代码，以及一份针对本节下面作业问题的实验日志（一份记录你尝试过的所有事情的文档）。

### 7.2 TinyStories

我们将从一个非常简单的数据集（TinyStories；[R. Eldan et al. [1]]）开始，模型会训练得很快，我们可以看到一些有趣的行为。获取此数据集的说明在第 1 节。下面是这个数据集样子的一个例子。

> **示例 (tinystories_example)：TinyStories 的一个例子**
>
> Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, "Wow, that is a really amazing vase! Can I buy it?" The shopkeeper smiled and said, "Of course you can. You can take it home and show all your friends how amazing it is!" So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!

#### 7.2.1 超参数调整

我们会告诉你一些非常基本的起始超参数，并要求你为其他一些找到工作良好的设置。

**词汇表大小** 10000。典型词汇表大小在数万到数十万。你应该改变这个，看看词汇表和模型行为如何变化。

**上下文长度** 256。像 TinyStories 这样的简单数据集可能不需要长序列长度，但对于后面的 OpenWebText 数据，你可能想改变它。尝试改变它，看看对每次迭代运行时间和最终困惑度的影响。

**d_model** 512。这比许多小型 Transformer 论文中使用的 768 维稍小，但这会让事情更快。

**d_ff** 1344。这大约是 $\frac{8}{3} d_{model}$，同时是 64 的倍数，这对 GPU 性能有好处。

**RoPE theta 参数** $\Theta$ 10000。

**层数和头数** 4 层，16 头。加在一起，这将给出约 17M 个非嵌入参数，是一个相当小的 Transformer。

**处理的总 token 数** 327,680,000（你的批次大小 × 总步数 × 上下文长度应大致等于这个值）。

你应该做一些试错来为以下其他超参数找到好的默认值：`learning rate`、`learning rate warmup`、其他 AdamW 超参数 $(\beta_1, \beta_2, \varepsilon)$ 和 `weight decay`。你可以在 [D. P. Kingma et al. [22]] 中找到这些超参数的一些典型选择。

#### 7.2.2 组合起来

现在你可以把一切组合起来，方法是获取一个训练好的 BPE 分词器，对训练数据集分词，并在你编写的训练循环中运行它。**重要提示：**如果你的实现正确且高效，上述超参数应能在 1 块 B200 GPU 上产生大约 20-30 分钟的运行时间。如果你的运行时间长得多，请检查并确保你的数据加载、检查点或验证损失代码没有成为运行时间瓶颈，并确保你的实现正确地进行了批处理。

#### 7.2.3 调试模型架构的技巧

我们强烈建议你熟悉你 IDE 内置的调试器（如 VSCode/Zed），相比用 print 语句调试它会节省你的时间。如果你使用文本编辑器，你可以使用像 `ipdb` 这样的工具。调试模型架构时的其他一些好做法是：

* 开发任何神经网络架构时，一个常见的第一步是在单个 minibatch 上过拟合。如果你的实现正确，你应该能够快速把训练损失驱动到接近零。
* 在各种模型组件中设置调试断点，检查中间张量的形状以确保它们符合你的预期。
* 监控激活、模型权重和梯度的范数，以确保它们没有爆炸或消失。

> **问题 (learning_rate)：调整学习率（2 B200 小时）（3 分）**
>
> 学习率是最重要的调整超参数之一。以你训练的基础模型为基础，回答以下问题：
>
> **(a)** 对学习率进行超参数扫描并报告最终损失（如果优化器发散则记录发散）。
>
> **交付物**：与多个学习率相关的学习曲线。解释你的超参数搜索策略。
>
> **交付物**：一个在 TinyStories 上验证损失（每 token）至多为 1.45 的模型
>
> > **低资源提示：在 CPU 或 Apple Silicon 上训练几步**
> >
> > 如果你在 `cpu` 或 `mps` 上运行，你应该将处理的总 token 数减少到 40,000,000，这足以产生合理流畅的文本。你也可以将目标验证损失从 1.45 提高到 2.00。
> >
> > 用调整过的学习率在 M4 Max 芯片和 36 GB RAM 上运行我们的解决方案代码，我们使用批次大小 × 总步数 × 上下文长度 = 32 × 5000 × 256 = 40,960,000 个 token，在 `cpu` 上需要 1 小时 22 分钟，在 `mps` 上需要 36 分钟。在第 5000 步，我们达到 1.80 的验证损失。
> >
> > 一些额外提示：
> > * 使用 $N$ 个训练步时，我们建议调整余弦学习率衰减调度，使其衰减在恰好第 $N$ 步终止（即达到最小学习率）。
> > * 使用 `mps` 时，**不要**使用 TF32 内核，即**不要**设置 `torch.set_float32_matmul_precision('high')`，你在 `cuda` 设备上可能会这样做。我们尝试用 `mps` 启用 TF32 内核（`torch` 版本 2.9.0），发现后端有时会使用静默损坏的内核，导致训练不稳定。
> > * 你可以通过用 `torch.compile` JIT 编译你的模型来加速训练。具体来说：
> >   * 在 `cpu` 上，用 `model = torch.compile(model)` 编译你的模型
> >   * 在 `mps` 上，你可以用 `model = torch.compile(model, backend="aot_eager")` 稍微优化反向传递。截至 `torch` 版本 2.9.0，`mps` 不支持用 Inductor 编译。
>
> **(b)** 民间智慧认为最佳学习率"处于稳定性的边缘"。研究学习率发散点如何与你的最佳学习率相关。
>
> **交付物**：学习率递增的学习曲线，其中包括至少一次发散运行，以及一份关于这如何与收敛率相关的分析。

现在让我们改变批次大小，看看训练会发生什么。批次大小很重要——它们让我们通过做更大的矩阵乘法从 GPU 获得更高的效率，但我们总是想要大批次大小是真的吗？让我们运行一些实验来找出答案。

> **问题 (batch_size_experiment)：批次大小变化（1 B200 小时）（1 分）**
>
> 将你的批次大小从 1 一路变化到 GPU 内存限制。尝试至少几个中间的批次大小，包括像 64 和 128 这样的典型大小。
>
> **交付物**：不同批次大小运行的学习曲线。如果必要，学习率应重新优化。
>
> **交付物**：几句话讨论你关于批次大小及其对训练影响的发现。

有了你的解码器，我们现在可以生成文本了！我们将从模型生成，看看它有多好。作为参考，你应该得到至少和下面例子一样好的输出。

> **示例 (ts_generate_example)：TinyStories 语言模型的采样输出**
>
> Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom. Lily's mom made a big pot of soup for dinner. Lily was so happy and said, "Thank you, Mommy! I love you." She helped her mom pour the soup into a big bowl. After dinner, Lily's mom made some yummy soup. Lily loved it! She said, "Thank you, Mommy! This soup is so yummy!" Her mom smiled and said, "I'm glad you like it, Lily." They finished cooking and continued to cook together. The end.

> **低资源提示：在 CPU 或 Apple Silicon 上生成文本**
>
> 如果你转而使用处理 40M token 的低资源配置，你应该看到仍然像英语但没有上面那么流畅的生成。例如，我们在 40M token 上训练的 TinyStories 语言模型的采样输出如下：
>
> Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head. One day, Sue went for a walk and met a ladybug! They became good friends and played on the path together.
> "Hey, Polly! Let's go out!" said Tim. Sue looked at the sky and saw that it was difficult to find a way to dance shining. She smiled and agreed to help the talking!"
> As Sue watched the sky moved, what it was. She

这里是精确的问题陈述以及我们的要求：

> **问题 (generate)：生成文本（1 分）**
>
> 使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要操纵解码器参数（温度、top-p 等）以获得流畅的输出。
>
> **交付物**：至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），以及一段关于此输出流畅度的简短评论和至少两个影响此输出好坏的因素。

### 7.3 消融和架构修改

理解 Transformer 的最好方法是实际修改它并观察它的行为。我们现在将做几个简单的消融和修改。

**消融 1：层归一化**

人们常说层归一化对 Transformer 训练的稳定性很重要。但也许我们想活得危险一点。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

> **问题 (layer_norm_ablation)：移除 RMSNorm 并训练（0.5 B200 小时）（1 分）**
>
> 从你的 Transformer 中移除所有 RMSNorm 并训练。在之前的最优学习率下会发生什么？你能通过使用更低的学习率获得稳定性吗？
>
> **交付物**：移除 RMSNorm 训练时的学习曲线，以及最佳学习率的学习曲线。
>
> **交付物**：几句关于 RMSNorm 影响的评论。

现在让我们研究另一个乍看之下似乎随意的层归一化选择。**Pre-norm** Transformer 块定义为

$$z = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)) \tag{25}$$
$$y = z + \text{FFN}(\text{RMSNorm}(z)). \tag{26}$$

这是对原始 Transformer 架构为数不多的"共识"修改之一，原始架构使用 **post-norm** 方法：

$$z = \text{RMSNorm}(x + \text{MultiHeadSelfAttention}(x)) \tag{27}$$
$$y = \text{RMSNorm}(z + \text{FFN}(z)). \tag{28}$$

让我们回到 post-norm 方法，看看会发生什么。

> **问题 (pre_norm_ablation)：实现 post-norm 并训练（0.5 B200 小时）（1 分）**
>
> 将你的 pre-norm Transformer 实现修改为 post-norm 的。用 post-norm 模型训练，看看会发生什么。
>
> **交付物**：post-norm Transformer 的学习曲线，与 pre-norm 的对比。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融 2：位置嵌入**

我们接下来将研究位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带 RoPE）与完全不包含位置嵌入（NoPE）。事实证明，仅解码器 transformer，即像我们已实现的那样带因果掩码的 transformer，理论上可以在不被显式提供位置嵌入的情况下推断相对或绝对位置信息 [Y.-H. H. Tsai et al., 2019；A. Kazemnejad et al., 2023]。我们现在将经验性地测试 NoPE 相比 RoPE 的表现。

> **问题 (no_pos_emb)：实现 NoPE（0.5 B200 小时）（1 分）**
>
> 将你带 RoPE 的 Transformer 实现修改为完全移除位置嵌入信息，看看会发生什么。
>
> **交付物**：一条比较 RoPE 和 NoPE 性能的学习曲线。

**消融 3：SwiGLU vs. SiLU**

接下来，我们将遵循 [N. Shazeer [20]] 并测试前馈网络中门控的重要性，方法是比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络的性能：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \text{SiLU}(W_1 x). \tag{29}$$

回忆一下，在我们的 SwiGLU 实现中，我们将内层前馈层的维度设为约 $d_{ff} = \frac{8}{3} d_{model}$（同时确保 $d_{ff} \bmod 64 = 0$，以利用 GPU 张量核）。在这个消融基线中，你的 $\text{FFN}_{\text{SiLU}}$ 实现应转而设 $d_{ff} = 4 \times d_{model}$，以大致匹配默认 SwiGLU 前馈网络（它有三个而非两个权重矩阵）的参数量。

> **问题 (swiglu_ablation)：SwiGLU vs. SiLU（0.5 B200 小时）（1 分）**
>
> **交付物**：一条比较 SwiGLU 和 SiLU 前馈网络性能的学习曲线，参数量大致匹配。
>
> **交付物**：几句话讨论你的发现。

> **低资源提示：GPU 资源有限的在线学生应在 TinyStories 上测试修改**
>
> 在作业的剩余部分，我们将转向一个更大规模、更嘈杂的网页数据集（OpenWebText），实验架构修改并（可选地）向课程排行榜提交。
>
> 在 OpenWebText 上训练 LM 到流畅需要很长时间，所以我们建议 GPU 访问有限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

我们现在将转向一个更标准的、从网页爬取创建的预训练数据集。OpenWebText [A. Gokaslan et al., 2019] 的一个小样本也作为单个文本文件提供：见第 1 节了解如何访问此文件。

这里是 OpenWebText 的一个例子。注意文本如何更真实、更复杂、更多样。你可能想通读训练数据集，以了解网页爬取语料库的训练数据是什么样子。

> **示例 (owt_example)：OWT 的一个例子**
>
> Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan Judge.
>
> Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, "no corner of American culture is more precisely counted, more passionately quantified, than performances of baseball players." With a few clicks here and there, you can find out that Noah Syndergaard's fastball revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the game's highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that seem ripped from a video game or science fiction novel. The rising ocean of data has empowered an increasingly important actor in baseball's culture: the analytical hobbyist.
>
> That empowerment comes with added scrutiny – on the measurements, but also on the people and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash that accompanies quantitative imperfection. He also knew the site's catching metrics needed to be reworked, and that it would take a learned mind – someone who could tackle complex statistical modeling problems – to complete the job.
>
> "He freaks us out." Harry Pavlidis
>
> Pavlidis had a hunch that Judge "got it" based on the latter's writing and their interaction at a site-sponsored ballpark event. [...]

> **注意**：对于这个实验，你可能需要重新调整你的超参数，如学习率或批次大小。

> **问题 (main_experiment)：在 OWT 上的实验（2 B200 小时）（2 分）**
>
> 使用与 TinyStories 相同的模型架构和总训练迭代次数，在 OpenWebText 上训练你的语言模型。它表现如何？
>
> **交付物**：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 损失的差异——我们应如何解释这些损失？
>
> **交付物**：来自 OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。这段文本的流畅度如何？为什么即使我们有和 TinyStories 相同的模型和计算预算，输出质量却更差？

### 7.5 你自己的修改 + 排行榜

恭喜你走到这一步。你几乎完成了！你现在将尝试改进 Transformer 架构，看看你的超参数和架构与班上其他学生相比如何。

**排行榜规则**

除了以下规则外没有其他限制：

**运行时间：** 你的提交在一块 B200 上最多运行 45 分钟。如果你使用 SLURM 或 Modal，你可能想在你的提交脚本中强制执行这一点。

**数据：** 你只能使用我们提供的 OpenWebText 训练数据集。

除此之外，你可以随心所欲地做任何事。

如果你在寻找实现什么的想法，你可以查看以下一些资源：

* 最先进的开源 LLM 家族，如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
* NanoGPT 竞速仓库（`github.com/KellerJordan/modded-nanogpt`），社区成员在其中发布许多有趣的小规模语言模型预训练"竞速"修改。例如，一个可追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定在一起（见 [A. Vaswani et al. [8]]（第 3.4 节）和 [A. Chowdhery et al. [16]]（第 2 节））。如果你尝试权重绑定，你可能需要减小嵌入/LM 头初始化的标准差。

在尝试完整的 45 分钟运行之前，你会想在 OpenWebText 的一个小子集或 TinyStories 上测试这些。

作为一个警告，我们确实要指出，你在这个排行榜中发现效果良好的一些修改可能不会泛化到更大规模的预训练。我们将在课程的缩放定律单元中进一步探讨这个想法。

> **问题 (leaderboard)：排行榜（10 B200 小时）（6 分）**
>
> 你将在上述排行榜规则下训练一个模型，目标是在 0.75 B200 小时内最小化你的语言模型的验证损失。
>
> **交付物**：记录的最终验证损失，一条清楚地显示挂钟时间 x 轴小于 45 分钟的相关学习曲线，以及一份你所做工作的描述。我们期望排行榜提交至少击败 5.0 损失的朴素基线。提交到排行榜：`github.com/stanford-cs336/assignment1-basics-leaderboard`。

---

## 参考文献

[1] R. Eldan and Y. Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?," 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, "OpenWebText corpus," 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," in *Proc. of ACL*, 2016.

[4] C. Wang, K. Cho, and J. Gu, "Neural Machine Translation with Byte-Level Subwords," 2019.

[5] P. Gage, "A new algorithm for data compression," *C Users Journal*, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, "Language Models are Unsupervised Multitask Learners," 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, "Improving Language Understanding by Generative Pre-Training," 2018.

[8] A. Vaswani et al., "Attention is All you Need," in *Proc. of NeurIPS*, 2017.

[9] T. Q. Nguyen and J. Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention," in *Proc. of IWSLT*, 2019.

[10] R. Xiong et al., "On Layer Normalization in the Transformer Architecture," in *Proc. of ICML*, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, "Layer Normalization," 2016.

[12] H. Touvron et al., "LLaMA: Open and Efficient Foundation Language Models," 2023.

[13] B. Zhang and R. Sennrich, "Root Mean Square Layer Normalization," in *Proc. of NeurIPS*, 2019.

[14] A. Grattafiori et al., "The Llama 3 Herd of Models," [Online]. Available: https://arxiv.org/abs/2407.21783

[15] A. Yang et al., "Qwen2.5 Technical Report," *arXiv preprint arXiv:2412.15115*, 2024.

[16] A. Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways," 2022.

[17] D. Hendrycks and K. Gimpel, "Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units," 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning," [Online]. Available: https://arxiv.org/abs/1702.03118

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, "Language Modeling with Gated Convolutional Networks," [Online]. Available: https://arxiv.org/abs/1612.08083

[20] N. Shazeer, "GLU Variants Improve Transformer," 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021.

[22] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. of ICLR*, 2015.

[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. of ICLR*, 2019.

[24] T. B. Brown et al., "Language Models are Few-Shot Learners," in *Proc. of NeurIPS*, 2020.

[25] J. Kaplan et al., "Scaling Laws for Neural Language Models," 2020.

[26] J. Hoffmann et al., "Training Compute-Optimal Large Language Models," 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, "The Curious Case of Neural Text Degeneration," in *Proc. of ICLR*, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel," in *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, "The Impact of Positional Encoding on Length Generalization in Transformers," in *Thirty-seventh Conference on Neural Information Processing Systems*, 2023. [Online]. Available: https://openreview.net/forum?id=Drrl2gcjzl
