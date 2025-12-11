# Music Recommendation Experiments

Experimenting with recommendation algorithms using a subset of the Spotify Million Playlist Dataset.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data

Download a subset of the Spotify Million Playlist Dataset from Kaggle:
- https://www.kaggle.com/datasets/himanshuwagh/spotify-million

Place the downloaded files in `data/raw/`.

## Project Structure

```
├── data/
│   ├── raw/          # Original dataset files
│   └── processed/    # Cleaned/transformed data
├── notebooks/        # Jupyter notebooks for exploration
├── src/              # Reusable Python modules
└── models/           # Saved model artifacts
```

## Notebooks

- `01_data_exploration.ipynb` - Initial data loading and EDA

