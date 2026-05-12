import axios from 'axios'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '',
  timeout: 30000,
})

// Retry on 502/503/504 (server cold-start or deploy bounce) with exponential backoff.
http.interceptors.response.use(null, async (error) => {
  const config = error.config
  if (!config) return Promise.reject(error)
  const status = error.response?.status
  const retryable = status === 502 || status === 503 || status === 504
  if (!retryable) return Promise.reject(error)
  config._retryCount = (config._retryCount || 0) + 1
  if (config._retryCount > 3) return Promise.reject(error)
  const delay = 500 * 2 ** (config._retryCount - 1)
  await new Promise(r => setTimeout(r, delay))
  return http(config)
})

export const api = {
  health:         ()                          => http.get('/health'),
  capabilities:   ()                          => http.get('/api/capabilities'),
  configs:        ()                          => http.get('/configs'),
  experiments:    (policy)                    => http.get('/experiments', { params: policy ? { policy } : {} }),
  simulate:       (body)                      => http.post('/simulate', body),
  simulateWizard: (body)                      => http.post('/simulate-wizard', body, { timeout: 90000 }),
  uploadEssData:  (formData)                  => http.post('/upload-ess-data', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  }),
  designSimulation: (body)                    => http.post('/design-simulation', body, { timeout: 60000 }),
  status:         (expId)                     => http.get(`/status/${expId}`),
  results:        (expId)                     => http.get(`/results/${expId}`),
  interview:      (expId, agentId, question)  => http.post(`/interview/${expId}/${agentId}`, { question }),
  anchor:         (expId, question)           => http.post(`/anchor/${expId}`, { question }),
  inject:         (expId, eventType, payload) => http.post(`/inject/${expId}`, { event_type: eventType, payload }),
  incomplete:     ()                          => http.get('/incomplete'),
  humanEvalScenarios: (pid)                   => http.get('/human-eval/scenarios', { params: pid ? { PROLIFIC_PID: pid } : {} }),
  humanEvalRating: (body)                     => http.post('/human-eval/rating', body),
  humanEvalResults: ()                        => http.get('/human-eval/results'),
}

export function gini(values) {
  const n = values.length
  if (!n) return 0
  const s = [...values].sort((a, b) => a - b)
  const total = s.reduce((acc, v) => acc + v, 0)
  if (!total) return 0
  let sum = 0
  for (let i = 0; i < n; i++) sum += (2 * (i + 1) - n - 1) * s[i]
  return Math.max(0, sum / (n * total))
}

export function mean(values) {
  if (!values.length) return 0
  return values.reduce((a, b) => a + b, 0) / values.length
}
