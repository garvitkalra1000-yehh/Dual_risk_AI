import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchWorklist } from '../services/api';

export default function WorklistPage() {
  const [rows, setRows] = useState();
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ limit: 25, risk: '', tier: '' });
  const navigate = useNavigate();

  useEffect(() => {
    fetchWorklist(filters)
      .then(setRows)
      .catch(() => setError('Unable to load worklist'));
  }, [filters]);

  if (error) return <div className="state error">{error}</div>;
  if (!rows) return <div className="state">Loading worklist...</div>;

  return (
    <div className="page">
      <h2>Priority Worklist</h2>
      <div className="toolbar">
        <select value={filters.risk} onChange={(e) => setFilters((p) => ({ ...p, risk: e.target.value }))}>
          <option value="">All Risk Types</option>
          <option value="academic">Academic Risk</option>
          <option value="placement">Placement Risk</option>
        </select>
        <select value={filters.tier} onChange={(e) => setFilters((p) => ({ ...p, tier: e.target.value }))}>
          <option value="">All Tiers</option>
          <option value="Red">Red</option>
          <option value="Amber">Amber</option>
          <option value="Green">Green</option>
        </select>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Student</th><th>Department</th><th>Success Score</th><th>Academic Risk</th><th>Placement Risk</th><th>Priority</th><th>Trend</th><th>Recommended Action</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.student_id} onClick={() => navigate(`/students/${r.student_id}`)}>
                <td>{r.student_id}</td>
                <td>{r.department}</td>
                <td>{Number(r.success_score).toFixed(1)}</td>
                <td>{((r.academic_risk_prob || 0) * 100).toFixed(1)}%</td>
                <td>{((r.placement_risk_prob || 0) * 100).toFixed(1)}%</td>
                <td>{Number(r.priority_score).toFixed(1)}</td>
                <td>{r.trend}</td>
                <td>{Array.isArray(r.recommended_actions) ? r.recommended_actions[0]?.intervention_type || 'N/A' : 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <div className="state">No students matched the selected filters.</div>}
    </div>
  );
}
