import { useEffect, useState } from 'react';
import { fetchDataQuality, fetchMetrics, fetchWeights, recomputeWeights } from '../services/api';

const keys = ['academic', 'attendance', 'lms', 'placement', 'engagement_skills', 'feedback'];

export default function AdminPage() {
  const [quality, setQuality] = useState([]);
  const [metrics, setMetrics] = useState();
  const [weights, setWeights] = useState();
  const [msg, setMsg] = useState('');

  useEffect(() => {
    fetchDataQuality().then(setQuality);
    fetchMetrics().then(setMetrics);
    fetchWeights().then(setWeights);
  }, []);

  const onChangeWeight = (k, v) => setWeights((prev) => ({ ...prev, [k]: Number(v) / 100 }));

  const handleRecompute = async () => {
    try {
      const result = await recomputeWeights(weights);
      setMsg(result.message);
    } catch {
      setMsg('Recompute failed');
    }
  };

  return (
    <div className="page">
      <h2>Admin: Model & Data Quality</h2>
      <div className="grid-2">
        <section className="panel">
          <h3>Data Quality</h3>
          {quality.map((q) => (
            <div key={q.source} className="admin-item">
              <strong>{q.source}</strong>
              <div>Rows: {q.original_rows} | Duplicates: {q.duplicate_rows} | Invalid: {q.invalid_values}</div>
            </div>
          ))}
        </section>
        <section className="panel">
          <h3>Model Metrics</h3>
          <pre>{JSON.stringify(metrics?.models || {}, null, 2)}</pre>
          <h4>Fairness (Synthetic Demo)</h4>
          <pre>{JSON.stringify(metrics?.fairness || {}, null, 2)}</pre>
        </section>
      </div>
      <section className="panel">
        <h3>Success Score Weights</h3>
        {!weights && <p>Loading weights...</p>}
        {weights && keys.map((k) => (
          <label key={k} className="weight-row">
            <span>{k}</span>
            <input type="range" min="0" max="100" value={(weights[k] || 0) * 100} onChange={(e) => onChangeWeight(k, e.target.value)} />
            <span>{(((weights[k] || 0) * 100)).toFixed(0)}%</span>
          </label>
        ))}
        <button onClick={handleRecompute}>Recalculate Success Scores</button>
        {msg && <p className="note">{msg}</p>}
      </section>
    </div>
  );
}
