# Rerankers and two-stage retrieval, measured

A hands-on study of reranking in a retrieval pipeline. It starts from Pinecone's [rerankers notebook](https://github.com/pinecone-io/examples/blob/master/learn/generation/better-rag/00-rerankers.ipynb), updates it so it runs today, and adds an evaluation that measures what the reranker adds and what it costs.

**The pipeline:** 41,584 chunks of arXiv papers → OpenAI `text-embedding-3-small` → Pinecone serverless index → top 25 by vector search → Cohere `rerank-v4.0-fast` → top 3.

## Results

50 test questions, each with one known answer chunk. Run on 2026-10-12.

| | Vector search only | With reranking | Change |
|---|---|---|---|
| Answer chunk ranked first (hit@1) | 60% | **78%** | +18 points |
| Answer chunk in the top 3 (hit@3) | 82% | **92%** | +10 points |
| Answer chunk in the top 10 (hit@10) | 92% | 94% | +2 points |
| MRR | 0.718 | 0.849 | +0.131 |
| nDCG@3 | 0.734 | 0.866 | +0.132 |
| Found by vector search at all (hit@25) | 94% | same | the ceiling |

| Step | Median | 95th percentile |
|---|---|---|
| Embed the question (OpenAI) | 412 ms | 796 ms |
| Vector search (Pinecone) | 140 ms | 266 ms |
| Rerank 25 chunks (Cohere) | 455 ms | 2,743 ms |

### What the numbers say

- **The gain is largest at the top.** 12 questions gained first place and 3 lost it.
- **The reranker is close to its ceiling.** Vector search found 47 of the 50 answers, and reranking put 46 of those 47 in the top 3. Any further gain has to come from the first stage.
- **The misses show what vector search is weak at.** Three answers were not in the top 25 at all. One was a table of results where the question named a model. Keyword search would match that name exactly, which is the case for hybrid search.
- **hit@10 barely moved,** because a reranker only reorders what the first stage returned. It matters most when few chunks are passed to the LLM.
- **Reranking is not free.** It roughly doubled the median time per query (565 ms to 1,112 ms), and 4 of the 50 rerank calls took over 2 seconds.

### Limits

- **Small test set.** With 50 questions, one question is 2 points. The top-3 gain is 6 questions gained and 1 lost: a direction, not proof.
- **Synthetic questions.** An LLM wrote each question from its chunk, so the question shares words with it. That flatters vector search. Real user questions are vaguer.
- **One answer chunk per question.** Another chunk may answer the same question and still count as a miss, so the scores are a lower bound.
- **A few weak test items,** such as questions about a table or a figure.
- **The timings are from a laptop and a trial Cohere key.** They are not production latency.

## How the evaluation works

It is the last section of the notebook, **Measuring the Reranker**.

1. **Test set.** Pick random chunks of 500 characters or more. An LLM writes one question that each chunk answers, or replies `SKIP` for reference lists and tables. The chunk's ID is the right answer. The set is saved to `rerank_eval_set.json`, and later runs load that file, so results stay comparable.
2. **Run.** For each question: embed it, get the top 25 from Pinecone, rerank all 25 with Cohere. Record the answer chunk's position before and after, and the time each step took. Reranking all 25 once gives the result for every cutoff.
3. **Report.** hit@1/3/5/10, MRR and nDCG for both pipelines, hit@25 as the ceiling, how many answers moved up or down, latency per step, and tables of the biggest improvements, the regressions and the misses.

## What changed from Pinecone's original

| Change | Why |
|---|---|
| `embed()` read `batch["text"]` from a list of strings | It raised a `TypeError`. It now embeds the list it receives. |
| The retry loop in `embed()` never stopped after a success | Every batch was embedded five times. It now returns on success and raises after five failed tries. |
| `!pip` → `%pip`, and `pyarrow==20.0.0` pinned | `%pip` installs into the notebook's own environment. `datasets==2.14.5` fails with newer pyarrow. |
| `text-embedding-ada-002` → `text-embedding-3-small` | Current embedding model, same 1,536 dimensions |
| `rerank-english-v2.0` → `rerank-v4.0-fast` | Current Cohere reranker |
| Index region `us-west-2` → `us-east-1` | |
| New section: **Measuring the Reranker** | The original compares two queries by eye. |
| The delete-index line is commented out | The index is reused between sessions. |

## Run it yourself

You need Python 3.10 and API keys for OpenAI, Pinecone and Cohere.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then open `00-rerankers.ipynb` in Jupyter or VS Code.

- **Keys.** The notebook reads `OPENAI_API_KEY`, `PINECONE_API_KEY` and `COHERE_API_KEY` from the environment. If one is not set, it asks for it. Keys are never saved in the notebook.
- **First run.** Run it from top to bottom. The embed-and-upsert cell is the slow, paid step: it embeds all 41,584 chunks. My run took 65 minutes, but that was before the retry fix, so expect it to be several times faster.
- **Later runs.** Skip the embed-and-upsert cell. The index keeps the vectors.
- **The evaluation** takes about 8 minutes with a Cohere trial key, because trial keys allow 10 rerank calls a minute. With a production key, set `RERANK_PAUSE_S = 0`.
- **The test set.** Keep `rerank_eval_set.json` to use my 50 questions; its chunk IDs match the dataset. Delete the file to have a new set written. `QUESTION_MODELS` lists the chat models to try for that.
- **When you are done,** uncomment `pc.delete_index(index_name)` in the last code cell.

## Files

| File | What it is |
|---|---|
| `00-rerankers.ipynb` | The notebook, with the outputs of my run |
| `rerank_eval_set.json` | The test set: 50 questions, each with its answer chunk's ID and text |
| `rerank_eval_results.json` | Per question: the answer chunk's position before and after reranking, and the timings |
| `requirements.txt` | The package versions I ran it with |

## What I would try next

- Hybrid search (keyword plus vectors) for the answers the first stage missed.
- `EVAL_TOP_K = 50`, to see whether the ceiling rises.
- A second reranker on the same 50 questions, for a like-for-like comparison.
- A metadata filter in the query. The chunks already store the paper's category and date.

## Credits and license

- **Pinecone** wrote the original notebook and the chapter it belongs to, [Rerankers and Two-Stage Retrieval](https://www.pinecone.io/learn/series/rag/rerankers/). Both are worth reading first.
- **Dataset:** [`jamescalam/ai-arxiv-chunked`](https://huggingface.co/datasets/jamescalam/ai-arxiv-chunked) on Hugging Face.
- **License:** MIT, like Pinecone's examples. `LICENSE` keeps Pinecone's copyright notice.
