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
  // Required fields
  year: number;
  month: number;
  day: number;
  hour: number;
  
  // Recommended forecast fields
  O3_forecast?: number | null;
  NO2_forecast?: number | null;
  T_forecast?: number | null;
  q_forecast?: number | null;
  u_forecast?: number | null;
  v_forecast?: number | null;
  w_forecast?: number | null;
  blh_forecast?: number | null; // Boundary layer height (new, recommended)
  
  // Optional satellite data
  NO2_satellite?: number | null;
  HCHO_satellite?: number | null;
  ratio_satellite?: number | null;
  
  // Optional solar features (auto-computed if not provided)
  SZA_deg?: number | null; // Solar zenith angle (auto-computed from site coordinates)
  solar_declination_deg?: number | null;
  hour_angle_deg?: number | null;
  solar_time_hr?: number | null;
  EoT_min?: number | null;
  
  // Optional lag features (new models use lag1, lag2, lag3, lag6, lag12)
  O3_target_lag1?: number | null;
  O3_target_lag2?: number | null;
  O3_target_lag3?: number | null;
  O3_target_lag6?: number | null;
  O3_target_lag12?: number | null;
  NO2_target_lag1?: number | null;
  NO2_target_lag2?: number | null;
  NO2_target_lag3?: number | null;
  NO2_target_lag6?: number | null;
  NO2_target_lag12?: number | null;
  
  // Legacy fields (not used by new models, kept for compatibility)
  NO2_sat_flag?: number | null;
  HCHO_sat_flag?: number | null;
  ratio_sat_flag?: number | null;
  O3_target_lag24?: number | null;
  O3_target_lag168?: number | null;
  NO2_target_lag24?: number | null;
  NO2_target_lag168?: number | null;
  
  // Site ID (for unified model)
  site_id?: number | null;
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
  // Note: HCHO_target, CO_target, PM25_target, PM10_target are not returned by the new models
  // but kept for backward compatibility
  HCHO_target?: number;
  CO_target?: number;
  PM25_target?: number;
  PM10_target?: number;
}

export interface PredictResponse {
  success: boolean;
  site_id: number | null;
  forecast_hours: number;
  predictions: Prediction[];
  message: string;
  live_source?: {
    station_id?: number;
    station_name?: string;
    station_location?: [number, number];
    measurement_time?: string;
    timezone?: string;
    overall_aqi?: number;
    observed_no2?: number;
    observed_o3?: number;
  };
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

export interface HistoricalDataPoint {
  year: number;
  month: number;
  day: number;
  hour: number;
  O3_observed: number | null;
  NO2_observed: number | null;
  datetime: string;
}

export interface HistoricalDataResponse {
  success: boolean;
  site_id: number;
  time_period: string;
  data_points: number;
  data: HistoricalDataPoint[];
}
