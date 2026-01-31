# Music Recommendation System

A controllable music recommendation system built with Track2Vec embeddings and an interactive 3D visualization web app.

## Overview

This project explores music recommendation using the Spotify Million Playlist Dataset. It implements:

- **Track2Vec**: Word2Vec-style embeddings trained on playlist co-occurrence (tracks that appear in the same playlists are similar)
- **Controllable Recommendations**: Tune parameters like popularity, artist diversity, and exploration
- **3D Visualization**: Interactive t-SNE visualization of 62K+ track embeddings
- **Web App**: Next.js application for exploring recommendations

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
│   └── export_data.py    # Export data for web app
└── web/                  # Next.js web application
```

## Setup

### Prerequisites

- Python 3.9+
- Node.js 18+ (for web app)

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data

Download the Spotify Million Playlist Dataset subset from Kaggle:
- https://www.kaggle.com/datasets/himanshuwagh/spotify-million

The notebooks expect data in `~/.cache/kagglehub/datasets/himanshuwagh/spotify-million/`.

## Notebooks

Run these in order:

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Load and explore the playlist dataset |
| `02_track_embeddings.ipynb` | Train Track2Vec embeddings using Word2Vec on playlist sequences |
| `03_spotify_features.ipynb` | Fetch artist metadata (genres, popularity) from Spotify API |
| `04_controllable_recommender.ipynb` | Build the ControllableRecommender class with tunable parameters |

## Web App

### Generate Data Files

First, export the data for the web app:

```bash
python src/export_data.py
```

This creates ~49MB of data files in `web/public/data/`.

### Run Locally

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000

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
