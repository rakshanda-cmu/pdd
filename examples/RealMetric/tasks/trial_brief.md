# Snaphomz – Data Science & AI/LLM Integration Intern
## Trial Assignment (48 hours) – Zero-Tolerance

**Goal:** Demonstrate end-to-end ability across data prep, a basic predictive model, and a minimal LLM feature—using only local/open-source tools.

### Data (Part A – Required)
- Load `/data/listings_sample.csv`.
- Clean: handle nulls, types, outliers; derive `price_per_sqft = price / sqft`.
- EDA: at least 3–5 charts/tables (by city, beds, baths, state).
- Output: concise insights (bullet points).

### Predictive (Part B – Required)
- Train a quick baseline:
  - Regression (target: `price_per_sqft`) **or**
  - Classification (3 price bands; you define thresholds).
- Show data split, metric(s), and a 2–3 line takeaway.

### LLM Feature (Part C – Required, pick one)
1) **RAG Q&A** over the `remarks` column:
   - Question example: “Does 123 Maple St have hardwood floors?”
   - Build a tiny retriever (e.g., bge-m3 embeddings or a keyword BM25 baseline).
   - Return answer **and** the supporting text span.
2) **Auto-summary**:
   - Generate a 2–3 sentence listing summary (address/price/beds/baths/remarks).
   - Use a local/open model (llama.cpp, mistral) **or** a clear rule/template. Explain your choice.

### Deliverables (All Required)
- `notebooks/trial.ipynb` – EDA + model + LLM feature (clean cells, runnable).
- `requirements.txt` or `environment.yml`.
- `README.md` – setup, decisions, how to run; limits/assumptions.
- `results/one_pager.md` or `.pdf` – highlights + 1–2 screenshots.

### Constraints & Guidance
- **No paid external API keys required**. Prefer local models or open inference.
- Keep it reproducible; pin versions where possible.
- If you get stuck, document assumptions and proceed.

### Scoring (100)
- Data handling (25) • Model & metrics (20) • LLM feature (30) • Clarity/docs (15) • Code quality (10).

### Zero-Tolerance
- Non-runnable submission, missing deliverables, or unverifiable authorship → disqualified.

### Submission
- Preferred: GitHub repo + Release named `snaphomz-trial-{yourname}`.
- Or: Single drive link (view/download enabled).
- Fill the submission form shared in the email.

Good luck!
