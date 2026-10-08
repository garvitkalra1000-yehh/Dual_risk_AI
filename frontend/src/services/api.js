import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
  timeout: 20000,
});

export const fetchSummary = async () => (await api.get('/summary')).data;
export const fetchStudents = async (params = {}) => (await api.get('/students', { params })).data;
export const fetchStudent = async (studentId) => (await api.get(`/students/${studentId}`)).data;
export const fetchWorklist = async (params = {}) => (await api.get('/worklist', { params })).data;
export const fetchSegments = async () => (await api.get('/segments')).data;
export const fetchMetrics = async () => (await api.get('/metrics')).data;
export const fetchDataQuality = async () => (await api.get('/data-quality')).data;
export const fetchWeights = async () => (await api.get('/weights')).data;
export const recomputeWeights = async (weights) => (await api.post('/weights/recompute', weights)).data;
export const startIntervention = async (payload) => (await api.post('/interventions', payload)).data;

export default api;
