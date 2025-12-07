/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

/**
 * Example response type for /api/demo
 */
export interface DemoResponse {
  message: string;
}

/**
 * Air Pollution Forecasting API Types
 */

export interface PredictInput {
  year: number;
  month: number;
  day: number;
  hour: number;
  O3_forecast: number;
  NO2_forecast: number;
  T_forecast: number;
  q_forecast: number;
  u_forecast: number;
  v_forecast: number;
  w_forecast: number;
  NO2_satellite?: number | null;
  HCHO_satellite?: number | null;
  ratio_satellite?: number | null;
  NO2_sat_flag?: number;
  HCHO_sat_flag?: number;
  ratio_sat_flag?: number;
  O3_target_lag1?: number | null;
  O3_target_lag24?: number | null;
  O3_target_lag168?: number | null;
  NO2_target_lag1?: number | null;
  NO2_target_lag24?: number | null;
  NO2_target_lag168?: number | null;
}

export interface PredictRequest {
  input_data: PredictInput[];
  site_id?: number;
  forecast_hours?: number;
}

export interface Prediction {
  year: number;
  month: number;
  day: number;
  hour: number;
  O3_target: number;
  NO2_target: number;
  HCHO_target: number;
  CO_target: number;
  PM25_target: number;
  PM10_target: number;
}

export interface PredictResponse {
  success: boolean;
  site_id: number;
  forecast_hours: number;
  predictions: Prediction[];
  message: string;
}

export interface LivePredictionResponse extends PredictResponse {
  live_source?: {
    station_id: number;
    station_name: string;
    station_location: [number, number];
    measurement_time: string;
    timezone: string;
    overall_aqi: number;
    observed_no2: number;
    observed_o3: number;
  };
}

export interface HealthResponse {
  success: boolean;
  message: string;
}

export interface ModelHealthResponse extends HealthResponse {
  models: {
    [key: string]: {
      available: boolean;
      last_updated?: string;
    };
  };
}

export interface ModelDetailResponse {
  success: boolean;
  site_id: number;
  model_name: string;
  features: string[];
  message: string;
}

export interface PollutantMetrics {
  rmse: number;
  mae: number;
  bias: number;
  r2: number;
  ria: number;
  baseline_rmse: number;
  rmse_reduction_pct: number;
}

export interface SiteMetrics {
  site_id: number;
  o3: PollutantMetrics;
  no2: PollutantMetrics;
}

export interface MetricsResponse {
  generated_at: string;
  sites: SiteMetrics[];
}
