import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import OverviewPage from './pages/OverviewPage';
import WorklistPage from './pages/WorklistPage';
import StudentDetailPage from './pages/StudentDetailPage';
import SegmentsPage from './pages/SegmentsPage';
import AdminPage from './pages/AdminPage';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/worklist" element={<WorklistPage />} />
          <Route path="/students/:studentId" element={<StudentDetailPage />} />
          <Route path="/segments" element={<SegmentsPage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
