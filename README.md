# Music Recommendation System

A controllable music recommendation system built with Track2Vec embeddings and an interactive 3D visualization web app.

## Overview

This project explores music recommendation using the Spotify Million Playlist Dataset. It implements:

- **Track2Vec**: Word2Vec-style embeddings trained on playlist co-occurrence (tracks that appear in the same playlists are similar)
- **Controllable Recommendations**: Tune parameters like popularity, artist diversity, and exploration
- **3D Visualization**: Interactive t-SNE visualization of 62K+ track embeddings
- **Web App**: Next.js application for exploring recommendations

The repo is in the middle of a transition from a notebook-first data science layout to a more standard code-and-tests workflow. The notebooks are still useful for exploration, but the scripted pipeline under `src/recommender_v2/` is the repeatable path for development.

## Demo

The web app allows you to:
1. Search for seed tracks
2. Adjust recommendation parameters with sliders
3. Explore the 3D embedding space (zoom, rotate, click to select)
4. Get personalized recommendations in real-time

## Project Structure

```
├── data/
│   ├── raw/              # Original dataset files
│   └── processed/        # Artist info with genres
├── models/
│   ├── track2vec.model   # Trained Word2Vec model
│   ├── track2vec.wordvectors
│   └── track_metadata.parquet
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Dataset EDA
│   ├── 02_track_embeddings.ipynb    # Train Track2Vec
│   ├── 03_spotify_features.ipynb    # Fetch artist metadata
│   └── 04_controllable_recommender.ipynb  # Recommender class
├── src/
│   ├── export_data.py    # Legacy one-shot export for the original web app
│   └── recommender_v2/   # Scripted collection / training / evaluation pipeline
└── web/                  # Next.js web application
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (for web app)

### Python Environment

```bash
# Create the local virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project in editable mode
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you prefer `make`, the same workflow is available via:

```bash
make venv
make install-dev
make test
```

If `python3.11` is not on your `PATH`, pass an explicit interpreter to the bootstrap target:

```bash
make venv PYTHON_BOOTSTRAP=/path/to/python3.11
```

`requirements.txt` is still available for the original notebook-oriented setup, but editable installs are the preferred path for ongoing development.

### Data

Download the Spotify Million Playlist Dataset subset from Kaggle:
- https://www.kaggle.com/datasets/himanshuwagh/spotify-million

The notebooks expect data in `~/.cache/kagglehub/datasets/himanshuwagh/spotify-million/`.

## Recommender V2 Pipeline

The repo now includes a scriptable training and evaluation pipeline under `src/recommender_v2/`.

Commands:

```bash
recommender-v2 collect_spotify --run-id my-run
recommender-v2 build_corpus --run-id my-run
recommender-v2 enrich_metadata --run-id my-run
recommender-v2 split_eval --run-id my-run
recommender-v2 train_retrieval --run-id my-run
recommender-v2 train_reranker --run-id my-run
recommender-v2 evaluate --run-id my-run
recommender-v2 export_web --run-id my-run
```

The module entry point still works too:

```bash
python -m src.recommender_v2 --help
```

Configuration lives in `config/recommender_v2.toml`.

Each run writes versioned artifacts under `data/runs/<run-id>/`, including raw JSONL payloads, normalized tables, eval splits, model artifacts, metrics, and exported web assets.

`collect_spotify` defaults to a local MPD fallback if Spotify credentials are not provided. For live Spotify collection, set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.

Optional enrichment:

- `LASTFM_API_KEY` enables Last.fm tag enrichment
- `MUSICBRAINZ_ENABLE=1` enables MusicBrainz release-year lookups

## Notebooks

The original notebooks are still useful for exploration, but they are no longer the required training path.

Run these in order if you want to revisit the exploratory workflow:

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Load and explore the playlist dataset |
| `02_track_embeddings.ipynb` | Train Track2Vec embeddings using Word2Vec on playlist sequences |
| `03_spotify_features.ipynb` | Fetch artist metadata (genres, popularity) from Spotify API |
| `04_controllable_recommender.ipynb` | Build the ControllableRecommender class with tunable parameters |

## Web App

### Generate Data Files

For the legacy app bundle, export the data with:

```bash
python src/export_data.py
```

For the V2 pipeline, `export_web` writes:

- full recommendation/search artifacts to `web/data/server/current/`
- visualization subset artifacts to `web/public/data/`

### Run Locally

```bash
npm --prefix web install
npm --prefix web run dev
```

Open http://localhost:3000

## Tests

The current automated coverage is a small smoke suite around the V2 pipeline:

```bash
make test
```

or directly:

```bash
python -m unittest discover -s tests -q
```

For a quick end-to-end model verification run, use the smaller dev config:

```bash
make verify-model
```

This defaults to `config/recommender_v2.dev.toml` and `RUN_ID=dev-verify`. For a full run or a specific existing run, override them:

```bash
make verify-model CONFIG=config/recommender_v2.toml RUN_ID=local-v3
```

Long-running retrieval, reranker, and evaluation commands now print stage-by-stage progress and also write status manifests under `data/runs/<run-id>/manifests/`.

### Deploy to Vercel

The web app is configured for Vercel deployment. Push to GitHub and import to Vercel.

## How It Works

### Track2Vec Embeddings

Tracks are embedded using Word2Vec trained on playlist "sentences":
- Each playlist is treated as a sentence of track IDs
- Tracks appearing in similar playlists get similar embeddings
- Results in 128-dimensional vectors for ~62K tracks

### Controllable Parameters

| Parameter | Effect |
|-----------|--------|
| **Popularity** | 0 = underground gems, 1 = mainstream hits |
| **Artist Diversity** | 0 = same artists OK, 1 = only new artists |
| **Exploration** | 0 = deterministic top picks, 1 = random sampling |
| **Genre Filter** | Include only specific genres |

### 3D Visualization

t-SNE reduces the 128-dim embeddings to 3D for visualization. Similar tracks cluster together, revealing genre neighborhoods and artist groupings.

## Tech Stack

- **ML**: gensim (Word2Vec), scikit-learn (t-SNE)
- **Data**: pandas, numpy, parquet
- **Web**: Next.js, React, TypeScript
- **Visualization**: Three.js, react-three-fiber
- **Styling**: Tailwind CSS

## License

MIT
