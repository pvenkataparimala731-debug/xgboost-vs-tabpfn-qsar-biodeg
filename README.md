# XGBoost vs TabPFN on qsar-biodeg

Companion Streamlit app for the MSc dissertation *"Benchmarking XGBoost
Against TabPFN Zero-Shot Predictions on TabArena Tabular Datasets: A
Feature-Structure Analysis of Foundation Model Underperformance."*

**Live demo:** _add your Streamlit Cloud URL here once deployed_

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — results dashboard, live XGBoost demo, dataset info |
| `requirements.txt` | Python dependencies for the app |
| `notebook.ipynb` | The full dissertation notebook (add this — see below) |

## 1. Push this to GitHub

```bash
# from inside this folder
git init
git add .
git commit -m "Initial commit: XGBoost vs TabPFN Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

If you already have a GitHub repo for this project, just copy `app.py`,
`requirements.txt`, and this `README.md` into it, then:

```bash
git add app.py requirements.txt README.md
git commit -m "Add Streamlit demo app"
git push
```

Don't forget to also add your dissertation notebook (e.g.
`Copy_of_A00075206_qsar_biodeg.ipynb`) to the repo — **first remove or
blank out the `TABPFN_TOKEN` line before pushing**, since that's a
personal API credential.

## 2. Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. Click **"New app"**, pick your repository, branch `main`, and set the
   main file path to `app.py`.
3. Click **Deploy**. The first build takes 2–5 minutes.

That's it — XGBoost trains live in the app using the real qsar-biodeg
dataset pulled from OpenML at runtime, so no model files need to be
committed.

## 3. (Optional) Enable live TabPFN

TabPFN needs a licensed API token, so it isn't installed by default and
the dashboard shows the dissertation's cached results instead of calling
TabPFN live. To enable live TabPFN predictions too:

1. Add `tabpfn` and `tabpfn-client` to `requirements.txt`.
2. In Streamlit Cloud, go to your app → **Settings → Secrets**, and add:
   ```toml
   TABPFN_TOKEN = "your-token-here"
   ```
3. Redeploy. `app.py` already checks for this secret automatically.

**Never commit your TabPFN token directly into the code or notebook.**

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
