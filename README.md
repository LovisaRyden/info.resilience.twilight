# Do we live in the Twilight Zone? – a study of information resilience and filters of perception in times of climate change.
**Scripts and configs for the master’s thesis**  
Indicators: sentiment, irony, trust/distrust, hostility, and misinformation-related categories (concept, belief, accusation).  
Models: multi-task RoBERTa fine-tune + Twitter-adapted RoBERTa (irony/sentiment specialist) merged in a hybrid.

> This repository ships **code and configuration only**. It does **not** include any raw user data or CCAM tables.

---

## What this repo contains
- **Scraping (YouTube)**
  - `Main_Scraper.py`
  - `Scraperforfirstyears.py`
- **Training & evaluation**
  - `finetune_multitask_roberta.py`
  - `evaluate_ft.py`
- **Inference & merge**
  - `merge_hybrid.py`
  - `prediction_meta.json`
  - `prediction_meta_hybrid.json`
- **Lexicon script**
  - `Sentiment_analysis_lexicon.py`
- **Configs**
  - `label_config.json`
  - `class_frequencies.json`
- **Documentation**
  - `Labelling guide.docx`

> This repository ships **code and configuration only**. It does **not** include raw comments or CCAM spreadsheets.

---

## Data policy (important)
- No raw YouTube comments, IDs, usernames, channel IDs, or URLs are included or redistributed.
- No CCAM spreadsheets are included. Year-specific toplines are available in the public *Climate Change in the American Mind* (CCAM) reports; cite by year in your thesis.
- Do not publish intermediate CSV/JSONL/NDJSON that contain `comment_id`, `video_id`, `channel_id`, or raw text.
- If you show examples in documentation, use **synthetic/paraphrased** text only.

---

## Intended use & limitations
- The models are designed for **aggregate, year-level analysis** of discourse (2010–2024).  
- They are **not suitable** for profiling or moderating individual users.  
- Rare labels (misinformation: concept, belief, accusation) were thresholded conservatively; treat them as **directional context** rather than precise counts.

---

## Environment
Create a clean environment and install the exact packages you used:
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> Generate `requirements.txt` from your working environment with:  
> `pip list --format=freeze > requirements.txt`

---

## Models
- **Base backbone (multi-task):** `roberta-base` (or the exact backbone you used for the base run).  
- **Specialist (irony/sentiment):** Twitter-adapted RoBERTa (CardiffNLP/TweetNLP lineage).  
- **Hybrid merge:** Take **irony + sentiment** from the specialist model; take **all other indicators** from the base model via `merge_hybrid.py`.

> The repository does **not** include model weights; the scripts fetch or expect them on first run. Use the same checkpoints listed in your thesis Methods.

---

## Pipeline (high level)
1. **Scrape (YouTube)**  
   - `Main_Scraper.py` (general), `Scraperforfirstyears.py` (bootstrap years).  
   - **API key:** set `YOUTUBE_API_KEY` as an environment variable.  
   - Output: year-bounded NDJSON/JSON with *no redistribution*.

2. **Fine-tune**  
   - `finetune_multitask_roberta.py` trains a shared encoder with:
     - 7 binary outputs: irony, trust, distrust, hostility, concept, belief, accusation.
     - 3-class sentiment (negative, neutral, positive).

3. **Evaluate**  
   - `evaluate_ft.py` computes metrics on the held-out test split and writes JSON reports under your run directory and/or `reports/`.

4. **Hybrid merge**  
   - `merge_hybrid.py` merges predictions from the base and the irony/sentiment specialist into one table with consistent columns.  
   - Writes `prediction_meta_hybrid.json` to document provenance.

---

## Repro tips
- Keep the provided directory structure; scripts use relative paths.
- Do not round rates early; keep full precision until plotting/reporting.
- For long texts, training used `max_length = 320` (see the training script).  
- Yearly analyses assume **sorted years** and consistent yearly caps to avoid volume artifacts.

---

## Ethics & platform terms
This code is for research. If you collect data:
- Comply with YouTube’s Terms of Service and local laws.
- Collect only public content; avoid storing personally identifiable information.
- Report only **aggregate** results (e.g., year-level proportions).
- Do not use the models to profile or penalize individuals.

---

## Citation
If you build on this repository, please cite the thesis and the upstream model/report sources. See `CITATION.cff`.

---

## License
MIT — see `LICENSE`.
