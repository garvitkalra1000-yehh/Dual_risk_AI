import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { fetchSegments } from '../services/api';

export default function SegmentsPage() {
  const [data, setData] = useState();

  useEffect(() => {
    fetchSegments().then(setData).catch(() => setData({ segments: [] }));
  }, []);

  if (!data) return <div className="state">Loading segments...</div>;

  const chartData = (data.segments || []).map((s) => ({ segment: s.segment, size: s.size }));

  return (
    <div className="page">
      <h2>Student Segments</h2>
      <section className="panel">
        <h3>Segment Sizes (K={data.selected_k})</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData}>
            <XAxis dataKey="segment" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="size" fill="#6366f1" />
          </BarChart>
        </ResponsiveContainer>
      </section>
      <div className="grid-2">
        {(data.segments || []).map((s) => (
          <section className="panel" key={s.cluster}>
            <h3>{s.segment}</h3>
            <p>{s.size} students ({s.percentage.toFixed(1)}%)</p>
            <p>{s.recommended_playbook}</p>
          </section>
        ))}
      </div>
    </div>
  );
}
