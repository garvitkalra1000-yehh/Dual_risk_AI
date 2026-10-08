export default function KpiCard({ label, value, tooltip }) {
  return (
    <div className="kpi" title={tooltip || label}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  );
}
