import {
  PredictRequest,
  PredictResponse,
  LivePredictionResponse,
  HealthResponse,
  ModelHealthResponse,
  ModelDetailResponse,
  MetricsResponse,
} from "@shared/api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface ApiError {
  message: string;
  status?: number;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    try {
      const url = `${this.baseUrl}${endpoint}`;
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        let errorMessage = `API Error: ${response.statusText} (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.detail || errorData.message) {
            errorMessage = errorData.detail || errorData.message;
          }
        } catch {
          // If response is not JSON, use the status text
        }
        throw {
          message: errorMessage,
          status: response.status,
        } as ApiError;
      }

      return await response.json();
    } catch (error) {
      if (error && typeof error === 'object' && 'message' in error) {
        throw error;
      }
      if (error instanceof Error) {
        // Network errors, CORS errors, etc.
        throw {
          message: `Network error: ${error.message}. Make sure the backend server is running on ${this.baseUrl}`
        } as ApiError;
      }
      throw { message: 'Unknown error occurred' } as ApiError;
    }
  }

  /**
   * Health check - basic API status
   */
  async healthCheck(): Promise<HealthResponse> {
    return this.request("/api/v1/health/");
  }

  /**
   * Check which models are available
   */
  async healthCheckModels(): Promise<ModelHealthResponse> {
    return this.request("/api/v1/health/models");
  }

  /**
   * Check specific model details for a site
   */
  async healthCheckModelDetail(siteId: number): Promise<ModelDetailResponse> {
    return this.request(`/api/v1/health/models/${siteId}`);
  }

  /**
   * Predict for a specific site
   */
  async predictSite(
    siteId: number,
    request: PredictRequest
  ): Promise<PredictResponse> {
    return this.request(`/api/v1/predict/site/${siteId}`, {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * Live prediction using WAQI feed for a specific site
   */
  async livePredictionSite(siteId: number): Promise<LivePredictionResponse> {
    return this.request(`/api/v1/predict/site/${siteId}/live`);
  }

  /**
   * Predict using unified model
   */
  async predictUnified(request: PredictRequest): Promise<PredictResponse> {
    return this.request("/api/v1/predict/unified", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * Auto-detect model and predict
   */
  async predict(request: PredictRequest): Promise<PredictResponse> {
    return this.request("/api/v1/predict/", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  /**
   * Get model evaluation metrics
   */
  async getMetrics(): Promise<MetricsResponse> {
    return this.request("/api/v1/predict/metrics");
  }
}

const apiClient = new ApiClient();
export default apiClient;
