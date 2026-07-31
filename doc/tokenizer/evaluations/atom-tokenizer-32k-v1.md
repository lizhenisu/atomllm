# Atom Tokenizer 32K 技术评测报告

> 评测版本：`tokenizer-technical-eval-a11b0effc3b8`  
> 生成时间：2026-07-29T05:56:48+00:00  
> Tokenizer SHA-256：`a95891ad6684bd437cbf1dfd2b3c82d2b24911bd69dd4a3bd20c04022a00c969`

## 摘要

本报告在 2,900 篇真实文档组成的合并评测集上评测 Atom 32K Byte-level BPE。样本覆盖最终 100B 配方的 20 个公开来源，共 16,105,667 字符、17,223,908 UTF-8 字节。

32K Tokenizer 的总体压缩率为 **3.0324 字符/Token**（分层 Bootstrap 95% CI 2.8648–3.2003），未知 Token 数和 NFC 往返失败数均为 **0**。相对 48K 候选，32K 在同一测试集上增加 **3.57%** Token，但词表与模型参数更少。

## 1. 评测对象

| 项目 | 32K 正式版本 | 48K 对照版本 |
| --- | --- | --- |
| 词表规模 | 32,000 | 48,000 |
| Tokenizer 文件 | 2.25 MiB | 3.44 MiB |
| 算法 | byte_level_bpe | byte_level_bpe |
| Unicode 规范化 | NFC | NFC |
| 未知 Token | `<unk>` | `<unk>` |

## 2. 数据与方法

质量评测使用 Tokenizer 数据快照中按每个完整来源最低 SHA-256 分数固定保留的 100 篇文档；这些文档在构造时即从训练集排除。19 个来源包括简体中文通用文本、中英文 Wikipedia、英文教育网页、数学、科学论文及12 种代码子集。核心指标为字符/Token、UTF-8 字节/Token、未知 Token 率和 NFC 规范化后的 encode→decode 往返一致性。

最终 100B 预训练配方后续加入 DCLM Baseline 补足英文预算。该来源未参与本 Tokenizer 训练，因此从固定 Parquet 分片按记录 ID 最低 SHA-256 抽取 1,000 篇，并与原有 19 来源 held-out 合并计算总体、语言、内容类型、来源及 32K/48K 对照指标。

置信区间使用按来源分层的 2,000 次非参数 Bootstrap，避免长文档或单一来源主导不确定性估计。质量统计使用多个 CPU 进程并行编码。

## 3. 核心质量结果

| 指标 | 32K 结果 |
| --- | ---: |
| 文档数 | 2,900 |
| 字符数 | 16,105,667 |
| Token 数 | 5,311,201 |
| 字符/Token | 3.032396 |
| UTF-8 字节/Token | 3.242940 |
| 文档级字符/Token P05 / P50 / P95 | 1.553 / 3.419 / 4.515 |
| 未知 Token | 0 |
| NFC 往返失败 | 0 |
| held-out 观测词表覆盖 | 31,034 / 32,000 (96.98%) |

### 3.1 按语言

| 语言 | 文档 | 字符 | Token | 字符/Token | 95% CI | 32K 相对 48K Token 增幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 1,200 | 8,085,155 | 2,981,342 | 2.7119 | 2.4997–2.9565 | 3.21% |
| en | 1,400 | 7,408,006 | 1,931,405 | 3.8356 | 3.7731–3.8934 | 3.65% |
| zh-Hans | 300 | 612,506 | 398,454 | 1.5372 | 1.5067–1.5727 | 5.86% |

### 3.2 按内容类型

| 内容类型 | 文档 | 字符/Token | 字节/Token | 未知 Token | 往返失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 1,200 | 2.7119 | 2.7326 | 0 | 0 |
| encyclopedia | 200 | 2.8755 | 3.8101 | 0 | 0 |
| general | 1,300 | 3.4945 | 3.9735 | 0 | 0 |
| math | 100 | 3.3890 | 3.4168 | 0 | 0 |
| science | 100 | 4.5653 | 4.5760 | 0 | 0 |

### 3.3 按公开数据源

| 来源 | 语言 | 内容 | 文档 | 字符/Token | 字节/Token |
| --- | --- | --- | ---: | ---: | ---: |
| [CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ) | zh-Hans | general | 100 | 1.5447 | 4.2188 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | en | general | 100 | 4.3217 | 4.3353 |
| [IndustryCorpus2](https://huggingface.co/datasets/BAAI/IndustryCorpus2) | zh-Hans | general | 100 | 1.5967 | 4.2475 |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | en | math | 100 | 3.3890 | 3.4168 |
| [peS2o v3](https://huggingface.co/datasets/allenai/peS2o) | en | science | 100 | 4.5653 | 4.5760 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (c) | code | code | 100 | 2.7308 | 2.7317 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (cpp) | code | code | 100 | 2.9846 | 3.0008 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (css) | code | code | 100 | 2.6849 | 2.6872 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (go) | code | code | 100 | 2.7170 | 2.7239 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (html) | code | code | 100 | 2.6582 | 2.7027 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (java) | code | code | 100 | 3.3924 | 3.4316 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (javascript) | code | code | 100 | 3.4311 | 3.4350 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (python) | code | code | 100 | 3.3315 | 3.3348 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (rust) | code | code | 100 | 3.4011 | 3.4046 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (shell) | code | code | 100 | 2.5781 | 2.5815 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (sql) | code | code | 100 | 2.1318 | 2.1630 |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) (typescript) | code | code | 100 | 3.4312 | 3.4370 |
| [DCLM Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) | en | general | 1,000 | 3.8729 | 3.8867 |
| [Wikipedia (en)](https://huggingface.co/datasets/wikimedia/wikipedia) | en | encyclopedia | 100 | 3.9099 | 3.9209 |
| [Wikipedia (zh)](https://huggingface.co/datasets/wikimedia/wikipedia) | zh-Hans | encyclopedia | 100 | 1.4292 | 3.6552 |

## 4. 鲁棒性探针

| 探针 | 样本 | Token | 未知 Token | NFC 往返失败 |
| --- | ---: | ---: | ---: | ---: |
| whitespace | 3 | 27 | 0 | 0 |
| symbols_and_emoji | 3 | 62 | 0 | 0 |
| identifiers | 3 | 55 | 0 | 0 |
| mixed_code | 3 | 45 | 0 | 0 |

## 5. 结论与限制

- 32K 正式 Tokenizer 在覆盖最终 100B 配方全部 20 个来源的合并评测集上实现零未知 Token、零 NFC 往返失败。
- 相比 48K，32K 平均多使用 3.57% Token；这是较小词表与模型侧效率之间的明确取舍。
- 中文压缩率应优先看字符/Token，英文与代码可同时参考字符/Token和字节/Token。

## 6. 复现

```bash
source .venv/bin/activate
python -m atomllm.tokenizer.technical_evaluation \
  --config configs/tokenizer/technical-evaluation-32k-v1.yaml \
  --overwrite
```

完整机器可读结果位于 `artifacts/tokenizer-technical-evaluations/atom-tokenizer-32k-v1/report.json`。
