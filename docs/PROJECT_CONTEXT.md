# Project Context & Technical Architecture: Hoops Name Generator

## 1. Project Overview & Goals

- **Project Name:** Hoops Name Generator (AI-Powered NBA Player Name Generator)
- **Core Intent:** Evolved from a monolithic Python script into a modular, tested, full-stack AI web application.
- **Primary Functionality:** Generates novel, phonetically authentic NBA player names with a conditional character-level LSTM trained on every player in NBA history (1940s–present, 5,063 cleaned names). Names that already belong to a real player are never returned.
- **Architecture Style:** Modular OOP, single-page frontend with in-browser model inference (deployed serverless on GitHub Pages), plus an optional stateless FastAPI REST backend.
- **Live Demo:** https://egemenilerigelen.github.io/hoops-name-generator/

## 2. Technical Stack & Backend Architecture

- **Core Technologies:** Python 3.10+, PyTorch, FastAPI, Uvicorn, Vanilla JavaScript (in-browser inference). Data collection uses `nba_api`; tests use `pytest` + `httpx2` + Node.js; deployment uses GitHub Actions and GitHub Pages.

### Data Pipeline
`train.py` orchestrates the steps; the modules themselves are pure and do no file I/O (`src/io_utils.py` handles JSON reading and writing).

1. `src/scraper.py` (`fetch_player_names()`) fetches all-time player names via `nba_api` → `data/raw/nba_players_raw.json`.
2. `src/preprocessing.py` (`NameCleaner`) converts Unicode to Latin (`Dončić` → `doncic`), lowercases, keeps only `a-z`, space and hyphen → `data/processed/clean_names.json`.
3. `src/data_loader.py` splits each name into **(first, last)**: generational suffixes (`jr`, `ii`, `iii`…) and middle initials are dropped, surname particles (`van`, `de`, `del`…) are kept (`dick van arsdale` → `dick` / `van arsdale`).

### Vocabulary (`Vocabulary` in `src/data_loader.py`)
- Reserved special tokens: `PAD=0`, `BOS_FIRST=1`, `BOS_LAST=2`, `EOS=3`; characters follow from index 4.
- Two distinct start tokens condition one network on *first name* vs *last name*.

### Neural Network (`src/lstm_model.py`)
- **Class:** `CharLSTM` (`torch.nn.Module`)
  1. `nn.Embedding(vocab_size, 32, padding_idx=PAD)`
  2. `nn.LSTM(32, 128, num_layers=2, dropout=0.2, batch_first=True)`
  3. `nn.Linear(128, vocab_size)`
- `forward(x, hidden=None)` returns logits for **every** timestep plus the hidden state.

### Training (`src/trainer.py`, entry point `train.py`)
- **`Trainer` class:** receives the model, data loaders, `TrainConfig` and device via its constructor (dependency injection), runs `fit()` and returns a `TrainingResult`. It knows nothing about files or data preparation, so tests reuse it with a tiny model.
- **`run_training()`:** composes the pipeline: load names → build `Vocabulary`/`NameDataset` → `build_loaders()` → `create_model()` → `Trainer.fit()` → `Checkpoint.save()`.
- **Sequence form:** input `[BOS] c₁ … cₙ` → target `c₁ … cₙ [EOS]` (teacher forcing over the full name, not fixed windows).
- **Loss:** Cross-entropy with `ignore_index=PAD`.
- **Setup:** Adam (lr 3e-3), batch 64, gradient clipping 1.0, seed 42, 90/10 train/validation split.
- **Early stopping:** up to 60 epochs, patience 6; the best-validation epoch is saved (current: val loss ≈ 1.74).

### Checkpoint (`src/checkpoint.py` → `models/lstm_weights.pth`)
- The `Checkpoint` dataclass is the **only** place that knows the on-disk format (`model_state`, `vocab_chars`, `model_config`, `known_names`, `version`). `save()`, `load()` and `build_model()` are used by training, inference and web export alike.
- Loaded with `torch.load(..., weights_only=True)` (no arbitrary pickle execution).
- Inference never reads the data files, so vocabulary indices cannot drift from the weights.

### Inference (`src/generator.py` — `NameGeneratorService`)
- **Stateful sampling:** the LSTM hidden state is carried across characters, so each prediction sees the whole prefix.
- **Temperature-based softmax sampling:** `P(c_i) = exp(z_i / T) / Σ_j exp(z_j / T)`
  - Low (0.2–0.4): low entropy, safe and common structures.
  - High (0.8–1.2): high entropy, novel and unusual combinations.
- **Validity & novelty filter:** special tokens are masked, words must be ≥ 2 chars, full names ≤ 24 chars (fits the UI card), and any name matching a real player (including suffix-free forms like *Gary Payton* for *Gary Payton II*) is resampled. Up to 50 attempts, then `NameGenerationError`.
- Roughly 10 ms per name on CPU.

### Backend API (`app.py`)
| Endpoint | Behavior |
|---|---|
| `GET /` | Serves the `static/` frontend (index.html, name-generator.js, model.json). |
| `GET /api/generate?temperature=0.7` | Returns `{"name": str, "temperature": float}`. `temperature` is validated to `0.1–2.0` → `422` otherwise; `503` if no novel name is found. |
| `GET /api/health` | `{"status": "ok"}` |
| `GET /docs` | Auto-generated OpenAPI docs. |

- `create_app(checkpoint_path, static_dir)` is an application factory; the model is loaded once per process via FastAPI `lifespan`.
- Endpoints receive the generator through `Depends(get_generator)`, so tests can swap in a fake via `app.dependency_overrides` (e.g. to exercise the 503 path).
- **Stateless:** no user state or database, so the service scales horizontally.
- No CORS middleware: the UI is served from the same origin.

### Configuration (`src/config.py`)
- All paths are resolved from the project root (the app runs from any working directory).
- `ModelConfig`, `TrainConfig`, generation limits and temperature bounds live in one place.

## 3. Directory Structure

```text
hoops-name-generator/
├── app.py                  # FastAPI app: UI + /api endpoints
├── train.py                # Pipeline entry: (--scrape) → clean → train → save
├── static/
│   ├── index.html          # Single-file HTML5/CSS3/Vanilla JS frontend
│   ├── name-generator.js   # In-browser LSTM inference
│   └── model.json          # Exported weights, vocab, limits, known names
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths, hyperparameters, generation limits
│   ├── io_utils.py         # read_json / write_json
│   ├── scraper.py          # fetch_player_names() via nba_api
│   ├── preprocessing.py    # NameCleaner: Unicode → Latin, cleaning
│   ├── data_loader.py      # split_name, Vocabulary, NameDataset, collate_batch
│   ├── lstm_model.py       # CharLSTM nn.Module
│   ├── checkpoint.py       # Checkpoint dataclass: on-disk format, create_model()
│   ├── trainer.py          # Trainer class + run_training() pipeline
│   ├── generator.py        # NameGeneratorService: sampling + novelty filter
│   └── web_export.py       # Checkpoint → static/model.json
├── data/
│   ├── raw/nba_players_raw.json
│   └── processed/clean_names.json
├── models/
│   └── lstm_weights.pth    # Self-contained checkpoint
├── tests/                  # data_loader, generator, trainer/checkpoint, api, web_model (+ js_runner.cjs)
├── .github/workflows/
│   └── pages.yml           # CI: test → GitHub Pages deploy
├── docs/
│   └── PROJECT_CONTEXT.md  # This document
├── .gitattributes          # LF line endings, binary *.pth
├── pytest.ini
├── requirements.txt        # Runtime: fastapi, uvicorn, torch (pinned)
├── requirements-dev.txt    # + nba_api, pytest, httpx2
└── README.md
```

## 4. Frontend & UX Design (`static/index.html`)

- **Tech Stack:** HTML5, modern CSS (Flexbox, CSS variables), Vanilla JavaScript (ES6+ async/await, Fetch API).
- **Aesthetic:** Retro basketball theme (dark wood court gradients, leather accents, JetBrains Mono / Anton / Oswald typography).
- **Key UX & UI features:**
  - Dynamic font scaling (`adjustFontSize()`) by name length to prevent layout shift.
  - Fixed card height (540px) to prevent vertical jumping between views.
  - Creativity gauge: range slider bound to `temperature` (0.20–1.20, default 0.70), always inside the API's accepted range.
  - Favorited names ("Roster") persisted in `localStorage` (`hoops_favorites`), with no database.
  - "Under the Hood" glassmorphism modal explaining the LSTM, character generation and temperature sampling.

## 5. Engineering Principles & Concepts Covered

- **Separation of concerns:** config, I/O, data, model definition, checkpoint format, training, inference service, API and UI are separate modules; pure logic (`NameCleaner`, `Trainer`, `NameGeneratorService`) does no file I/O.
- **Loose coupling / dependency injection:** classes receive their collaborators through constructors (`Trainer`, `NameGeneratorService`), the API resolves the generator via `Depends`, and only `Checkpoint` knows the serialization format. Stateless helpers stay plain functions instead of being forced into classes.
- **Frontend contract:** the UI talks to the model only through `HoopsGenerator.generateName(temperature)` (browser) or `GET /api/generate` (server).
- **Reproducibility:** pinned dependencies, seeded training, deterministic validation split, self-contained checkpoints.
- **Model lifecycle / MLOps:** offline training with pre-trained inference; early stopping on a validation set; distinction from continuous/online learning (data drift, concept drift, canary deployments, Kubeflow/Kafka pipelines).
- **PyTorch & serialization:** `state_dict` checkpoints, safe loading with `weights_only=True`.
- **Testing:** pytest suite covering preprocessing, name splitting, vocabulary/dataset shapes, generator novelty and limits, API validation (including a faked failing generator), Trainer/Checkpoint round-trips, and JS ↔ PyTorch numerical parity.
- **Git workflow:** atomic, conventional commits (`feat`, `refactor`, `test`, `build`, `docs`).

## 6. How to Run

```bash
# 1. Create & activate a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Start the server, then open http://127.0.0.1:8000
python -m uvicorn app:app --reload
```

Optional:

```bash
pip install -r requirements-dev.txt   # data collection + tests
python -m pytest                      # run tests
python train.py                       # retrain (~1 min on CPU) + re-export model.json
python train.py --scrape              # re-fetch NBA data, then retrain
python -m src.generator               # print sample names in the terminal
```

## 7. Deployment (GitHub Pages, serverless)

- **Why serverless:** the model is tiny (~220k parameters) but PyTorch is large, and container hosting with free tiers is unreliable (e.g. Hugging Face Docker Spaces require a paid PRO plan). Running inference in the browser makes hosting free, with no cold starts.
- **Export (`src/web_export.py`):** the checkpoint becomes `static/model.json` (~1.3 MB): vocabulary, special-token indices, model config, generation limits, known names, and every tensor as base64 little-endian float32 (bit-exact with PyTorch).
- **Browser inference (`static/name-generator.js`):** a JavaScript port of `NameGeneratorService` with an embedding lookup, a stacked LSTM cell (PyTorch gate order i, f, g, o), a linear head, temperature softmax sampling, the same validity/novelty filter, and title-casing. The model is downloaded once and then cached in memory; generation takes ~9 ms per name.
- **Frontend integration:** `index.html` calls `HoopsGenerator.generateName(temperature)` instead of `fetch('/api/generate')`. The visual design is unchanged.
- **Verification:** `tests/test_web_model.py` checks that `model.json` matches the checkpoint, that the JS forward pass matches PyTorch logits (atol 1e-4) via Node.js, and that JS-generated names are novel and fit the UI.
- **CI/CD (`.github/workflows/pages.yml`):** pytest on every push and PR; on `main`, `static/` is uploaded and deployed to GitHub Pages.
- **FastAPI** still serves the same `static/` folder plus the REST API for local use.
