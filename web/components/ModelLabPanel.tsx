'use client';

import type { ModelLabSnapshot } from '@/lib/modelLab';
import { formatPercent, formatSignedPercentDelta } from '@/lib/modelLab';

interface ModelLabPanelProps {
  snapshot: ModelLabSnapshot | null;
  isLoading: boolean;
  error: string | null;
}

const metricKeys = [
  'ndcg@10',
  'recall@50',
  'catalog_coverage@50',
  'unique_artist_coverage@50',
  'mean_popularity_percentile',
] as const;

const modelLabels: Record<string, string> = {
  legacy_track2vec: 'Legacy Track2Vec',
  retrieval: 'V2 Retrieval',
  reranker: 'Reranker',
};

export default function ModelLabPanel({ snapshot, isLoading, error }: ModelLabPanelProps) {
  if (isLoading) {
    return (
      <section className="bg-spotify-gray/30 rounded-xl p-6 text-spotify-light">
        Loading model lab...
      </section>
    );
  }

  if (error) {
    return (
      <section className="bg-spotify-gray/30 rounded-xl p-6">
        <h2 className="text-white text-xl font-bold mb-2">Model Lab</h2>
        <p className="text-red-300">{error}</p>
      </section>
    );
  }

  if (!snapshot) {
    return (
      <section className="bg-spotify-gray/30 rounded-xl p-6">
        <h2 className="text-white text-xl font-bold mb-2">Model Lab</h2>
        <p className="text-spotify-light">
          No model lab snapshot found. Run `make export-web CONFIG=config/recommender_v2.toml RUN_ID=local-v3`.
        </p>
      </section>
    );
  }

  const retrieval = snapshot.scorecard.find((row) => row.model === 'retrieval');
  const reranker = snapshot.scorecard.find((row) => row.model === 'reranker');
  const rerankerPromotion =
    snapshot.promotion.reranker_promoted === true
      ? 'Promoted'
      : snapshot.promotion.reranker_promoted === false
        ? 'Blocked'
        : 'Not confirmed';

  return (
    <section className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <SummaryCard label="Run" value={snapshot.run.id} detail={snapshot.run.status} />
        <SummaryCard label="Tracks" value={snapshot.data.tracks.toLocaleString()} detail="catalog rows" />
        <SummaryCard label="Playlists" value={snapshot.data.playlists.toLocaleString()} detail="training source" />
        <SummaryCard
          label="Reranker"
          value={rerankerPromotion}
          detail={snapshot.run.live ? 'live collection' : snapshot.run.source}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-spotify-gray/30 rounded-xl p-6">
          <div className="flex items-center justify-between gap-4 mb-4">
            <div>
              <h2 className="text-white text-xl font-bold">Test Set Scorecard</h2>
              <p className="text-spotify-light text-sm">
                Higher ranking metrics are good; coverage and popularity show recommendation shape.
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-spotify-gray text-spotify-light">
                  <th className="text-left py-3 pr-4 font-medium">Model</th>
                  {metricKeys.map((key) => (
                    <th key={key} className="text-right py-3 px-3 font-medium whitespace-nowrap">
                      {snapshot.metricLabels[key] ?? key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {snapshot.scorecard.length === 0 ? (
                  <tr>
                    <td colSpan={metricKeys.length + 1} className="py-6 text-center text-spotify-light">
                      No test scorecard rows available.
                    </td>
                  </tr>
                ) : (
                  snapshot.scorecard.map((row) => (
                    <tr key={row.model} className="border-b border-spotify-gray/50">
                      <td className="py-3 pr-4 text-white font-medium">{modelLabels[row.model] || row.model}</td>
                      {metricKeys.map((key) => (
                        <td key={key} className="py-3 px-3 text-right text-spotify-light">
                          {formatPercent(row.metrics[key])}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="bg-spotify-gray/30 rounded-xl p-6">
          <h2 className="text-white text-xl font-bold mb-4">What To Improve</h2>
          <div className="space-y-4">
            {snapshot.diagnostics.map((diagnostic) => (
              <div
                key={`${diagnostic.title}-${diagnostic.body}`}
                className={diagnostic.severity === 'warning' ? 'border-l-4 border-yellow-400 pl-4' : 'border-l-4 border-spotify-green pl-4'}
              >
                <h3 className="text-white font-semibold">{diagnostic.title}</h3>
                <p className="text-spotify-light text-sm mt-1">{diagnostic.body}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {retrieval && reranker && (
        <div className="bg-spotify-gray/30 rounded-xl p-6">
          <h2 className="text-white text-xl font-bold mb-4">Reranker vs Retrieval</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <DeltaCard
              label="Ranking lift"
              value={reranker.metrics['ndcg@10'] - retrieval.metrics['ndcg@10']}
              goodWhenPositive
            />
            <DeltaCard
              label="Recall lift"
              value={reranker.metrics['recall@50'] - retrieval.metrics['recall@50']}
              goodWhenPositive
            />
            <DeltaCard
              label="Coverage change"
              value={reranker.metrics['catalog_coverage@50'] - retrieval.metrics['catalog_coverage@50']}
              goodWhenPositive
            />
          </div>
        </div>
      )}
    </section>
  );
}

function SummaryCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="bg-spotify-gray/30 rounded-xl p-5">
      <div className="text-spotify-light text-sm">{label}</div>
      <div className="text-white text-2xl font-bold mt-1">{value}</div>
      <div className="text-spotify-light text-xs mt-2">{detail}</div>
    </div>
  );
}

function DeltaCard({
  label,
  value,
  goodWhenPositive,
}: {
  label: string;
  value: number;
  goodWhenPositive: boolean;
}) {
  const isGood = goodWhenPositive ? value >= 0 : value <= 0;
  const color = isGood ? 'text-spotify-green' : 'text-yellow-400';
  return (
    <div className="bg-spotify-black/40 rounded-lg p-4">
      <div className="text-spotify-light text-sm">{label}</div>
      <div className={`${color} text-2xl font-bold mt-1`}>{formatSignedPercentDelta(value)}</div>
    </div>
  );
}
