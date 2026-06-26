# Detecting Misinformation in the Age of LLMs: A Comparative Analysis

Code accompanying the UC3M Master's thesis (TFM) *Detecting Misinformation in the
Age of LLMs: A Comparative Analysis*. 

The study compares four detector families on **GossipCop++** (binary fake/real),
evaluated on a single held-out test set of **2,796** articles balanced across four
subclasses — human-fake (HF), machine-fake (MF), human-real (HR), machine-real
(MR):

- **Traditional ML** on TF-IDF / count features: Logistic Regression, Naive Bayes
  (Gaussian on TF-IDF *and* multinomial on counts), Random Forest
- **Fine-tuned `bert-base-uncased`**
- **Prompted LLMs**, zero-shot and 8-shot: GPT-4o-mini, Claude Haiku 4.5,
  Gemini 2.5 Flash

Every detector scores the **same** test rows and the **same** document text, so the
comparison is like-for-like. The random seed is **2026** throughout, and `"fake"`
is the positive class everywhere.

## Repository layout

```
analysis/
  01_data_prep.qmd            # load + clean + dedup + balance + 60/20/20 split → shared modelling table
  02_traditional_models.qmd   # train LR / NB×2 / RF, freeze predictions, evaluate
  03a_bert_finetune_gpu.py    # GPU fine-tuning script (run on Colab) → bert_test_predictions.csv
  03b_bert_eval.qmd           # build BERT input + evaluate BERT from frozen predictions
  04a_llm_infer_openai.qmd    # LLM inference — GPT-4o-mini  → llm_preds_gpt4o_mini_{zero,few}.csv
  04a_llm_infer_anthropic.qmd # LLM inference — Claude Haiku → llm_preds_claude_haiku_{zero,few}.csv
  04a_llm_infer_gemini.qmd    # LLM inference — Gemini Flash → llm_preds_gemini_flash_{zero,few}.csv
  04b_llm_eval.qmd            # parse LLM answers, evaluate, McNemar, full comparison
  API Keys.example.txt        # template for the LLM API keys (copy to "API Keys.txt")
  data/GossipCop++/           # raw dataset — download yourself (not in this repo)
artifacts/                    # committed frozen predictions, split assignments, exemplar IDs (no text)
results/                      # metrics tables (results/tables/), figures (results/figures/)
```

Each detector family writes **frozen per-row predictions** and its evaluation reads
those back, so the metrics reproduce without re-training. The committed prediction
files contain only `doc_id`, `subclass`, `truth`, `pred`/`raw` (+ `prob_fake` where
available) — **no article text**.

## Reproducing the results

The dataset is **not** included to make sure we don't violate copyright by uploading full article text. Download GossipCop++
yourself from the original repository: <https://github.com/mbzuai-nlp/Fakenews-dataset> (GossipCop++, from Su et al.).

Then place the four JSON files here:

```
analysis/data/GossipCop++/HF.json
analysis/data/GossipCop++/MF.json
analysis/data/GossipCop++/HR.json
analysis/data/GossipCop++/MR.json
```

Run order:

1. **`01_data_prep.qmd`** — needs the downloaded dataset. Builds the shared
   modelling table (`artifacts/gossip_model_data.csv`, gitignored — it holds text)
   and the 60/20/20 split.
2. **`02_traditional_models.qmd`** — trains the classical models, writes
   `artifacts/ml_preds_*.csv`, and evaluates them.
3. **`03b_bert_eval.qmd`** — folds BERT into the comparison. BERT itself is fine-tuned
   separately on a GPU: run the BERT modelling-data chunk locally to write
   `gossip_bert_data.csv`, then run `03a_bert_finetune_gpu.py` on Google Colab
   (GPU runtime) and drop the resulting `bert_test_predictions.csv` +
   `bert_training_log.csv` into `artifacts/`.
4. **`04b_llm_eval.qmd`** — folds in the LLMs.

**`03b` and `04b` reproduce from the committed predictions alone** — no dataset, no
GPU, no API access required. Only `01`/`02` need the dataset, and only the LLM
runners need API keys.

### Running the LLMs 

The six `artifacts/llm_preds_*.csv` files are committed, so `04b_llm_eval.qmd` runs
without any API calls. To regenerate them, copy `analysis/API Keys.example.txt` to
`analysis/API Keys.txt`, fill in your keys, and **Run All Chunks** in each
`04a_llm_infer_*.qmd`. The runners are resumable and safe to interrupt. Each writes its
two prediction files and is skipped automatically once they exist. To run 
`04a_llm_infer_*.qmd` files from scratch, you would need to delete prediction files and 
pay the API cost for each LLM.
