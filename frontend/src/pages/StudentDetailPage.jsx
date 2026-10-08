import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { fetchStudent, startIntervention } from '../services/api';

export default function StudentDetailPage() {
  const { studentId } = useParams();
  const [data, setData] = useState();
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const load = useCallback(
    () => fetchStudent(studentId).then(setData).catch(() => setError('Unable to load student details')),
    [studentId]
  );

  useEffect(() => {
    load();
  }, [load]);

  if (error) return <div className="state error">{error}</div>;
  if (!data) return <div className="state">Loading student details...</div>;

  const trendRows = (data.trends?.months || []).map((m, i) => ({
    month: m,
    academicRisk: (data.trends?.academic_risk?.[i] || 0) * 100,
    placementRisk: (data.trends?.placement_risk?.[i] || 0) * 100,
    success: data.trends?.success_score?.[i] || 0,
  }));

  const handleIntervention = async () => {
    const action = data.recommended_actions?.[0] || {
      intervention_type: 'General Guidance',
      owner: 'Faculty',
    };
    try {
      await startIntervention({
        student_id: data.student_id,
        intervention_type: action.intervention_type,
        owner: action.owner,
        status: 'Started',
      });
      setMessage('Intervention started. Follow-ups are SIMULATED.');
      load();
    } catch {
      setMessage('Failed to start intervention');
    }
  };

  return (
    <div className="page">
      <h2>Student Details: {data.student_id}</h2>
      <div className="grid-4">
        <div className="kpi"><div className="kpi-label">Department</div><div className="kpi-value">{data.department}</div></div>
        <div className="kpi"><div className="kpi-label">Semester</div><div className="kpi-value">{data.semester}</div></div>
        <div className="kpi"><div className="kpi-label">Success Score</div><div className="kpi-value">{Number(data.success_score).toFixed(1)}</div></div>
        <div className="kpi"><div className="kpi-label">Confidence</div><div className="kpi-value">{Number(data.score_confidence).toFixed(1)}%</div></div>
      </div>
      <section className="panel">
        <h3>Risk & Success Trends</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={trendRows}>
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="academicRisk" stroke="#ef4444" />
            <Line type="monotone" dataKey="placementRisk" stroke="#f59e0b" />
            <Line type="monotone" dataKey="success" stroke="#3b82f6" />
          </LineChart>
        </ResponsiveContainer>
      </section>
      <div className="grid-2">
        <section className="panel">
          <h3>Academic Risk Drivers</h3>
          <ul>{(data.explanations?.academic_top_drivers || []).map((x) => <li key={x.feature}>{x.feature}: {x.direction}</li>)}</ul>
          <h3>Placement Risk Drivers</h3>
          <ul>{(data.explanations?.placement_top_drivers || []).map((x) => <li key={x.feature}>{x.feature}: {x.direction}</li>)}</ul>
        </section>
        <section className="panel">
          <h3>Recommended Actions</h3>
          <ul>{(data.recommended_actions || []).map((a) => <li key={a.intervention_type}>{a.intervention_type} ({a.owner}, {a.deadline_days} days)</li>)}</ul>
          <button onClick={handleIntervention}>Start Intervention</button>
          {message && <p className="note">{message}</p>}
          <h4>Intervention History</h4>
          <ul>{(data.interventions || []).map((i, idx) => <li key={idx}>{i.intervention_type} - {i.status} ({i.start_date})</li>)}</ul>
          {(data.interventions || []).length === 0 && <p>No interventions yet.</p>}
        </section>
      </div>
    </div>
  );
}
