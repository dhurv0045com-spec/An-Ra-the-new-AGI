import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Services
export const getSystemStatus = async () => {
  const response = await api.get('/status');
  return response.data;
};

export const getBriefing = async () => {
  const response = await api.get('/briefing');
  return response.data;
};

export const getActiveGoals = async () => {
  const response = await api.get('/goals');
  return response.data;
};

export const spawnGoal = async (title, description, priority = 'medium') => {
  const response = await api.post('/goal', { title, description, priority });
  return response.data;
};

export const sendChatMessage = async (message) => {
  const response = await api.post('/chat', { message });
  return response.data;
};

export const triggerTraining = async () => {
  const response = await api.post('/train');
  return response.data;
};
