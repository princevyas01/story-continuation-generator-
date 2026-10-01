import {
  GenerateRequest,
  GenerateResponse,
  CompareRequest,
  CompareResponse,
  ModelStatusResponse,
  ModelMetricsResponse,
  HealthResponse,
} from '../types';

const API_BASE = '/api/v1';

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!res.ok) {
      let errData: any = {};
      try {
        errData = await res.json();
      } catch {
        errData = { detail: res.statusText };
      }
      const message = errData.detail || `Request failed with status ${res.status}`;
      throw new ApiError(res.status, message, errData);
    }

    return await res.json();
  } catch (err: any) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(0, err.message || 'Network connection failed. Is backend running?');
  }
}

export const api = {
  checkHealth: (): Promise<HealthResponse> => {
    return request<HealthResponse>('/health');
  },

  getModelStatus: (): Promise<ModelStatusResponse> => {
    return request<ModelStatusResponse>('/model/status');
  },

  getModelMetrics: (): Promise<ModelMetricsResponse> => {
    return request<ModelMetricsResponse>('/model/metrics');
  },

  generateStory: (req: GenerateRequest, signal?: AbortSignal): Promise<GenerateResponse> => {
    return request<GenerateResponse>('/generate', {
      method: 'POST',
      body: JSON.stringify(req),
      signal,
    });
  },

  compareGenerations: (req: CompareRequest): Promise<CompareResponse> => {
    return request<CompareResponse>('/generate/compare', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },
};
