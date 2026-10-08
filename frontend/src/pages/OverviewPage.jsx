import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { fetchSummary } from '../services/api';
import KpiCard from '../components/KpiCard';

const tierColors = { Green: '#22c55e', Amber: '#f59e0b', Red: '#ef4444', 'N/A': '#94a3b8' };

export default function OverviewPage() {
  const [data, setData] = useState();
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSummary().then(setData).catch(() => setError('Unable to load summary'));
  }, []);

  if (error) return <div className="state error">{error}</div>;
  if (!data) return <div className="state">Loading overview...</div>;

  const tierData = Object.entries(data.tier_counts || {}).map(([name, value]) => ({ name, value }));
  const deptData = Object.entries(data.department_comparison || {}).map(([department, score]) => ({ department, score }));

  return (
    <div className="page">
      <h2>Overview</h2>
      <div className="grid-4">
        <KpiCard label="Total Students" value={data.total_students} />
        <KpiCard label="Avg Success Score" value={data.avg_success_score?.toFixed?.(2) ?? data.avg_success_score} />
        <KpiCard label="Academic Risk Count" value={data.academic_risk_count} />
        <KpiCard label="Placement Risk Count" value={data.placement_risk_count} />
      </div>
      <div className="grid-2">
        <section className="panel">
          <h3>Success Tier Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={tierData} dataKey="value" nameKey="name" outerRadius={100} label>
                {tierData.map((entry) => (
                  <Cell key={entry.name} fill={tierColors[entry.name] || '#8884d8'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </section>
        <section className="panel">
          <h3>Department Comparison (Avg Success Score)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={deptData}>
              <XAxis dataKey="department" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Bar dataKey="score" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </section>
      </div>
    </div>
  );
}
