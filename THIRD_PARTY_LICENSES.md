# Third-party licenses and attribution

This replication is built directly on the original authors' released work. That
release is what makes the replication possible at all, and the attribution below
is not a formality.

---

## Lost in the Middle (Liu et al.)

**Repository:** https://github.com/nelson-liu/lost-in-the-middle
**License:** MIT, Copyright (c) 2023 Nelson Liu
**Paper:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele
Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models
Use Long Contexts.* Transactions of the Association for Computational
Linguistics (2024). https://arxiv.org/abs/2307.03172

### What this repository takes from it

| What | Where it lives here | Nature of the use |
|---|---|---|
| `normalize_answer`, `best_subspan_em` | `src/litm2026/scoring.py` | **Verbatim port.** Character-for-character. |
| Prompt templates (`qa.prompt`, `closedbook_qa.prompt`, `kv_retrieval.prompt`, and the two variants) | `src/litm2026/prompts/` | **Vendored unchanged.** |
| Prompt construction, including the `Document [{i}](Title: {t}) {text}` format string | `src/litm2026/prompting.py` | **Ported**, restructured for typing but semantically identical. |
| Multi-document QA data, key-value retrieval data | Downloaded at runtime, **not committed** | Used as released. |

The verbatim port is deliberate. A replication that changes the metric or the
prompt is not a replication — its numbers cannot be compared to the paper's.
Where this repository *does* deviate (the chat-format adaptation, principally),
that deviation is documented in `REPLICATION.md` and is switchable by a flag.

### Data

The data is **not redistributed in this repository** (it is ~257 MB, and it is
not mine to re-host). `scripts/download_data.py` fetches it from the authors'
repository. The data derives from Natural Questions and Wikipedia; see the
original repository for its provenance.

### Full MIT license text of the original work

```
MIT License

Copyright (c) 2023 Nelson Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## SQuAD evaluation script

The answer normalization inside `normalize_answer` originates upstream of the
Lost in the Middle release, in the SQuAD evaluation script:

https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/

It is reproduced here through the original authors' port of it.
