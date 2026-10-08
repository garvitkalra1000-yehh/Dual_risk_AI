import { NavLink } from 'react-router-dom';

const nav = [
  ['/', 'Overview'],
  ['/worklist', 'Worklist'],
  ['/segments', 'Segments'],
  ['/admin', 'Admin'],
];

export default function Layout({ children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>DualRisk AI</h1>
        <p>Smart Campus Success</p>
        <nav>
          {nav.map(([to, label]) => (
            <NavLink key={to} to={to} className={({ isActive }) => (isActive ? 'active' : '')} end={to === '/'}>
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
