export interface ModelLabMetricRow {
  model: string;
  split: 'test';
  metrics: {
    'ndcg@10': number | null;
    'recall@50': number | null;
    'catalog_coverage@50': number | null;
    'unique_artist_coverage@50': number | null;
    'mean_popularity_percentile': number | null;
  };
}

export interface ModelLabDiagnostic {
  severity: 'info' | 'warning';
  title: string;
  body: string;
}

export interface ModelLabSnapshot {
  schemaVersion: number;
  run: {
    id: string;
    status: 'evaluated' | 'incomplete';
    source: string;
    collector: string;
    live: boolean;
  };
  data: {
    playlists: number;
    tracks: number;
    artists: number;
    playlistTracks: number;
    splits: {
      train: number;
      val: number;
      test: number;
    };
  };
  models: {
    best_retrieval: {
      experiment: string | null;
      params: Record<string, number>;
      metrics: Record<string, number>;
      modelPath: string | null;
    };
  };
  promotion: Record<string, boolean | number | string | null>;
  scorecard: ModelLabMetricRow[];
  diagnostics: ModelLabDiagnostic[];
  metricLabels: Record<string, string>;
}

export async function fetchModelLabSnapshot(): Promise<ModelLabSnapshot | null> {
  const response = await fetch('/data/model_lab.json');
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to load model lab snapshot: ${response.status}`);
  }
  return response.json();
}

export function formatPercent(value: number | null): string {
  if (value === null) {
    return 'Unavailable';
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatSignedPercentDelta(value: number | null): string {
  if (value === null) {
    return 'Unavailable';
  }
  const sign = value >= 0 ? '+' : '';
  return `${sign}${(value * 100).toFixed(1)} pts`;
}
