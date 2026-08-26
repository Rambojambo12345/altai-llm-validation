# ALTAI/CDA LLM-assisted coding validation

Independent validation of the human coding in `master_components_final.csv`
(2,245 components across 60 AI-auditing frameworks) using an LLM as a
second, blinded coder against the codebook in `codebook.json`. See the
project docs for the full methodological write-up.

## Running this in a GitHub Codespace

A Codespace runs entirely in the cloud on GitHub's infrastructure — nothing
touches your own machine, and anyone with repo access (a co-author, a
reviewer, later a replication audience) can launch an identical environment
with one click. That's the reproducibility property you want for this.

### 1. Create the repository

- On github.com, click **New repository**. Name it something like
  `altai-llm-validation`. **Set it to Private** for now — the underlying
  data is your unpublished manuscript's dataset. You can make it public
  later as a replication package once the paper is out.
- Do not initialize with a README (you already have one here).

### 2. Get these files into the repository

Simplest path, no git installation needed: on the new repo's page, click
**Add file → Upload files**, then drag in everything from this folder
(keep the `.devcontainer` folder structure intact — GitHub's uploader
preserves folder paths when you drag a folder in). Commit directly to `main`.

If you're comfortable with git instead:
```
cd path/to/this/folder
git init
git add .
git commit -m "Initial LLM validation pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/altai-llm-validation.git
git push -u origin main
```

### 3. Add your API key as a Codespaces secret (do this before launching)

This keeps the key out of every file in the repo, so it's never committed
and never visible to anyone who later gets read access.

- In the repository, go to **Settings → Secrets and variables → Codespaces**.
- Click **New repository secret**.
- Name: `ANTHROPIC_API_KEY`
- Value: your key from console.anthropic.com (Settings → API Keys)
- Save.

GitHub automatically injects this as an environment variable into every
Codespace launched from this repo — nothing else to configure. The
`.devcontainer/devcontainer.json` in this repo needs no changes for that
to work.

### 4. Launch the Codespace

- On the repo's main page, click the green **Code** button → **Codespaces**
  tab → **Create codespace on main**.
- Wait for it to build (the container installs `requirements.txt`
  automatically via `postCreateCommand` — you'll see this happen in the
  terminal on first launch, takes roughly a minute).
- This opens a full VS Code environment in your browser, with a terminal
  at the bottom.

### 5. Run it — exact commands, in order, in the Codespace terminal

```bash
# sanity check first: verify the API key is visible and codes ~20 items
python run_llm_coding.py --dev-run
```

Check `output/coded_results.csv` looks sensible before spending on the full
run — confirm the codes look plausible and the API calls succeeded.

```bash
# the full production run: ~2,245 components x 3 runs each
python run_llm_coding.py
```

This takes a while (network-bound, one call at a time) and costs a few
dollars in API usage at current pricing. It writes everything to `output/`
as it goes, so if it's interrupted you can inspect what's there — note it
does not currently resume a partial run, so a full clean run is simplest.

```bash
# compute reliability statistics from the results
python compute_reliability.py
```

This prints Krippendorff's alpha, Cohen's kappa, per-class precision/
recall/F1, and self-consistency rates to the terminal and writes
`output/reliability_report.json`, `output/confusion_matrix_*.csv`, and
`output/adjudication_sample.csv` (150 human/LLM disagreements, blinded,
ready for you to adjudicate by hand — see the protocol doc for why this
step matters).

### 6. Get the results back out

Right-click any file in `output/` in the Codespace file explorer and choose
**Download**, or from the terminal:
```bash
git add output/coded_results.csv output/reliability_report.json output/run_manifest.json output/raw_runs.jsonl
git commit -m "Add validation run results"
git push
```
(`output/` is gitignored by default so test runs don't clutter the repo —
once you have a final run you're keeping, force-add those specific files as
above. `raw_runs.jsonl` is your full reproducibility log; keep it even
though it's large.)

### 7. Stop the Codespace when done

Codespaces bill by compute-hour while running. From github.com, go to
**your profile → Codespaces**, and either stop or delete it once you're
done — stopping preserves the environment for next time, deleting removes
it entirely. GitHub free tier includes a monthly quota that easily covers
this.

## What to send back

`output/reliability_report.json` plus `output/run_manifest.json` (the
exact model ID and parameters used) — that's what turns into the reliability
paragraph in the methods section.
