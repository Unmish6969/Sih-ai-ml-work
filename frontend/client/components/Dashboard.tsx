import React, { useState, useEffect } from 'react';
import Header from '@/components/Header';
import { RefreshCw, AlertCircle, Clock, MapPin, Car, CheckCircle2 } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';
import { 
  LivePredictionResponse, 
  ModelHealthResponse, 
  PredictResponse,
  ModelDetailResponse 
} from '@shared/api';
import apiClient from '@/services/apiClient';
import { Button } from '@/components/ui/button';
import PollutantGauge from '@/components/PollutantGauge';
import SiteSelector from '@/components/SiteSelector';
import ForecastChart from '@/components/ForecastChart';
import InteractiveForecastTool from '@/components/InteractiveForecastTool';
import HealthStatus from '@/components/HealthStatus';
import { ResponsiveLine } from '@nivo/line';
import DelhiAirMap from '@/components/DelhiAirMap';
type TabType = 'overview' | 'health' ;
type TimeRange = 1 | 24 | 48;

export default function Dashboard() {
  const { theme, toggleTheme } = useTheme();
  const [selectedSite, setSelectedSite] = useState(1);
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [timeRange, setTimeRange] = useState<TimeRange>(24);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeNav, setActiveNav] = useState('dashboard');
  
  // Data states - ONLY real API data
  const [liveData, setLiveData] = useState<LivePredictionResponse | null>(null);
  const [forecastData, setForecastData] = useState<PredictResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<ModelHealthResponse | null>(null);
  const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);
  
  // Gas selection states - ONLY O3 and NO2 (what API provides)
  const [selectedGas, setSelectedGas] = useState<string | null>(null);
  const [visibleGases, setVisibleGases] = useState<Set<string>>(new Set([
    'O₃ (ppb)',
    'NO₂ (ppb)',
  ]));

  // Fetch all data - ONLY REAL API DATA, NO DUMMY VALUES
  const fetchAllData = async (siteId: number, hours: TimeRange = 24) => {
    try {
      setLoading(true);
      setError(null);

      // Fetch ONLY from backend API - NO fallback to dummy data
      const liveData = await apiClient.livePredictionSite(siteId);
      const forecastData = await apiClient.livePrediction24hSite(siteId);
      const healthData = await apiClient.healthCheckModels();
      const modelData = await apiClient.healthCheckModelDetail(siteId);

      setLiveData(liveData);
      setForecastData(forecastData as unknown as PredictResponse);
      setHealthStatus(healthData);
      setModelDetail(modelData);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch data from backend';
      console.error('Error fetching data:', err);
      setError(errorMessage);
      
      // NO FALLBACK - show error instead of dummy data
      setLiveData(null);
      setForecastData(null);
      setHealthStatus(null);
      setModelDetail(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSiteChange = (siteId: number) => {
    setSelectedSite(siteId);
  };

  const handleRefresh = () => {
    fetchAllData(selectedSite);
  };

  // Update activeTab based on activeNav
  useEffect(() => {
    if (activeNav === 'dashboard') {
      setActiveTab('overview');
    }
  }, [activeNav]);

  // Initial fetch - NO CSV LOADING
  useEffect(() => {
    fetchAllData(selectedSite, timeRange);
  }, [selectedSite, timeRange]);

  // Generate trend data from API forecast data ONLY
  const generateTrendData = () => {
    if (!forecastData || !forecastData.predictions || forecastData.predictions.length === 0) {
      return []; // No dummy data
    }

    // Use actual forecast data from API
    return forecastData.predictions.map((pred) => {
      const hour = Math.floor(pred.hour || 0);
      const timeStr = `${String(hour).padStart(2, '0')}:00`;
      
      return {
        time: timeStr,
        O3: pred.O3_target || 0,
        NO2: pred.NO2_target || 0,
      };
    });
  };

  const trendData = generateTrendData();

  // Gas definitions - ONLY O3 and NO2 (what the backend provides)
  const gasDefinitions = [
    { id: 'O₃ (ppb)', key: 'O3', color: '#06b6d4', label: 'O₃' },
    { id: 'NO₂ (ppb)', key: 'NO2', color: '#2563eb', label: 'NO₂' },
  ];

  // Helper function to convert hex to rgba
  const hexToRgba = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Transform data for Nivo ResponsiveLine with click-to-highlight functionality
  const nivoData = gasDefinitions
    .filter(gas => visibleGases.has(gas.id))
    .map(gas => {
      const isSelected = selectedGas === gas.id;
      const isGrayedOut = selectedGas !== null && selectedGas !== gas.id;
      
      return {
        id: gas.id,
        data: trendData.map(d => ({ x: d.time, y: d[gas.key as keyof typeof d] as number })),
        color: isGrayedOut ? hexToRgba('#9ca3af', 0.3) : gas.color,
        originalColor: gas.color, // Keep original color for points
        lineWidth: isSelected ? 3 : 2,
        pointSize: isSelected ? 18 : 16, // Circular dots at every hour - consistent size for all gases
      };
    });

  // Handle gas selection toggle
  const handleGasToggle = (gasId: string) => {
    setVisibleGases(prev => {
      const newSet = new Set(prev);
      if (newSet.has(gasId)) {
        newSet.delete(gasId);
      } else {
        newSet.add(gasId);
      }
      return newSet;
    });
  };

  // Handle gas click for highlighting
  const handleGasClick = (gasId: string) => {
    if (selectedGas === gasId) {
      setSelectedGas(null); // Deselect if clicking the same gas
    } else {
      setSelectedGas(gasId);
    }
  };


  // Helper to safely format numeric values
  const fmt = (value: number | undefined | null, digits = 1) =>
    typeof value === 'number' ? value.toFixed(digits) : 'N/A';

  return (
    <div className={`flex flex-col flex-1 ${theme === 'dark' ? 'bg-slate-950 text-slate-100' : 'bg-white text-slate-900'}`}>
      <Header 
        title="Air Quality Overview"
        subtitle="Current gases levels across all 7 regions of Delhi."
        onRefresh={handleRefresh}
        loading={loading}
        showRefresh={true}
        theme={theme}
        onThemeToggle={toggleTheme}
      />
      
      {/* Content */}
      <div className={`p-8 ${theme === 'dark' ? 'bg-slate-950' : 'bg-slate-50'}`}>
          {error && (
            <div className={`mb-6 p-4 border rounded-lg flex items-start gap-3 ${theme === 'dark' ? 'bg-red-900/20 border-red-700' : 'bg-red-50 border-red-300'}`}>
              <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${theme === 'dark' ? 'text-red-500' : 'text-red-600'}`} />
              <div className={`text-sm ${theme === 'dark' ? 'text-red-200' : 'text-red-700'}`}>
                <p className="font-semibold">Error</p>
                <p className="mt-1">{error}</p>
              </div>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <RefreshCw className="w-8 h-8 animate-spin text-cyan-500 mx-auto mb-4" />
                <p className={theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}>Loading data...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Tab Content */}
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Current Air Pollution Data - ONLY O3 and NO2 from API */}
                  {liveData && (
                    <div className="space-y-4">
                      <h2 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>
                        Current Air Pollution Data (Live Predictions)
                      </h2>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* NO2 Card */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-blue-500/20' : 'bg-blue-50'}`}>
                              <Car className="w-6 h-6 text-blue-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>NO₂</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Nitrogen Dioxide</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-blue-400' : 'text-blue-600'}`}>
                                {liveData.predictions[0].NO2_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>

                        {/* O3 Card */}
                        <div className={`relative rounded-lg border p-5 shadow-sm transition-all hover:shadow-md ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-slate-50 border-slate-200'}`}>
                          <div className="absolute top-4 right-4">
                            <CheckCircle2 className="w-6 h-6 text-green-500" />
                          </div>
                          <div className="flex items-center gap-3 mb-3">
                            <div className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-cyan-500/20' : 'bg-cyan-50'}`}>
                              <Car className="w-6 h-6 text-cyan-500" />
                            </div>
                            <div>
                              <h3 className={`text-2xl font-bold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>O₃</h3>
                              <p className={`text-xs mt-0.5 ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Ozone</p>
                            </div>
                          </div>
                          <div className="mt-4">
                            <div className="flex items-baseline gap-2 mb-2">
                              <span className={`text-4xl font-bold ${theme === 'dark' ? 'text-cyan-400' : 'text-cyan-600'}`}>
                                {liveData.predictions[0].O3_target.toFixed(1)}
                              </span>
                              <span className={`text-sm font-medium ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>
                                µg/m³
                              </span>
                            </div>
                            <p className={`text-base font-semibold ${theme === 'dark' ? 'text-green-400' : 'text-green-600'}`}>Good</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Delhi Heatmaps */}
                  <Suspense fallback={
                    <div className={`rounded-xl border p-6 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                      <p className={`text-sm ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Loading map...</p>
                    </div>
                  }>
                    <MapErrorBoundary>
                      <DelhiAirMap onRefresh={() => fetchAllData(selectedSite, timeRange)} />
                    </MapErrorBoundary>
                  </Suspense>

                  {/* Time Range Selector & Trend Chart */}
                  <div className={`rounded-xl border p-6 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                    <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
                      <h2 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Pollution Trends</h2>
                      <div className="flex items-center gap-4 flex-wrap">
                        {/* Gas Selector */}
                        <div className={`flex items-center gap-2 flex-wrap ${theme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'} rounded-lg p-2`}>
                          <span className={`text-xs font-medium px-2 ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Gases:</span>
                          {gasDefinitions.map(gas => (
                            <label
                              key={gas.id}
                              className={`flex items-center gap-1.5 px-2 py-1 rounded cursor-pointer transition-all ${
                                visibleGases.has(gas.id)
                                  ? theme === 'dark' ? 'bg-slate-600' : 'bg-slate-200'
                                  : theme === 'dark' ? 'hover:bg-slate-600/50' : 'hover:bg-slate-200/50'
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={visibleGases.has(gas.id)}
                                onChange={() => handleGasToggle(gas.id)}
                                className="cursor-pointer"
                              />
                              <span
                                className="text-xs font-medium"
                                style={{ color: visibleGases.has(gas.id) ? gas.color : theme === 'dark' ? '#94a3b8' : '#64748b' }}
                              >
                                {gas.label}
                              </span>
                            </label>
                          ))}
                        </div>
                        {/* Time Range Selector */}
                        <div className={`flex gap-2 rounded-lg p-1 ${theme === 'dark' ? 'bg-slate-700' : 'bg-slate-100'}`}>
                          {[1, 24, 48].map((hours) => (
                            <button
                              key={hours}
                              onClick={() => setTimeRange(hours as TimeRange)}
                              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                                timeRange === hours
                                  ? 'bg-cyan-600 text-white'
                                  : theme === 'dark'
                                  ? 'text-slate-300 hover:bg-slate-600'
                                  : 'text-slate-600 hover:bg-slate-200'
                              }`}
                            >
                              {hours}h
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div style={{ height: '400px' }}>
                      <ResponsiveLine
                        data={nivoData}
                        margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
                        xScale={{ type: 'point' }}
                        yScale={{
                          type: 'linear',
                          min: 'auto',
                          max: 'auto',
                          stacked: false,
                          reverse: false
                        }}
                        curve="monotoneX"
                        colors={(d: any) => d.color}
                        lineWidth={((d: any) => {
                          if (!d || typeof d !== 'object') return 2;
                          const width = typeof d.lineWidth === 'number' && !isNaN(d.lineWidth) && d.lineWidth > 0 
                            ? d.lineWidth 
                            : 2;
                          return Math.max(1, Math.min(10, width)); // Clamp between 1 and 10
                        }) as any}
                        // Use original color for points (not grayed out)
                        axisTop={null}
                        axisRight={null}
                        axisBottom={{
                          tickSize: 5,
                          tickPadding: 5,
                          tickRotation: 0,
                          legend: 'Time',
                          legendOffset: 36,
                          legendPosition: 'middle',
                          tickValues: trendData.length > 12 
                            ? trendData.filter((_, i) => i % Math.ceil(trendData.length / 12) === 0).map(d => d.time)
                            : trendData.map(d => d.time)
                        }}
                        axisLeft={{
                          tickSize: 5,
                          tickPadding: 5,
                          tickRotation: 0,
                          legend: 'Concentration',
                          legendOffset: -50,
                          legendPosition: 'middle'
                        }}
                        enablePoints={true}
                        pointSize={6}
                        pointColor={(point: any) => {
                          // Find the series by matching the data structure - ensure each curve has its own color
                          const seriesId = point.serieId || point.seriesId || point.id;
                          const series = nivoData.find(s => s.id === seriesId);
                          
                          // First try to get originalColor from series (preserves color even when grayed out)
                          if (series && (series as any)?.originalColor) {
                            return (series as any).originalColor;
                          }
                          
                          // If no originalColor, try to get from gasDefinitions directly
                          const gasDef = gasDefinitions.find(g => g.id === seriesId);
                          if (gasDef) {
                            return gasDef.color;
                          }
                          
                          // Fallback to series color or point color
                          if (series) {
                            return series.color || '#94a3b8';
                          }
                          
                          return point.serieColor || point.seriesColor || point.color || '#94a3b8';
                        }}
                        pointBorderWidth={1.5}
                        pointBorderColor={(point: any) => {
                          // Use white border for dark theme, dark border for light theme - more prominent like reference
                          return theme === 'dark' ? '#ffffff' : '#1e293b';
                        }}
                        pointLabelYOffset={-12}
                        enablePointLabel={false}
                        isInteractive={true}
                        animate={true}
                        enableArea={false}
                        useMesh={false}
                        legends={[
                          {
                            anchor: 'bottom-right',
                            direction: 'column',
                            justify: false,
                            translateX: 100,
                            translateY: 0,
                            itemsSpacing: 4,
                            itemDirection: 'left-to-right',
                            itemWidth: 90,
                            itemHeight: 24,
                            itemOpacity: 0.75,
                            symbolSize: 14,
                            symbolShape: 'circle',
                            onClick: (datum) => handleGasClick(datum.id as string),
                            effects: [
                              {
                                on: 'hover',
                                style: {
                                  itemBackground: 'rgba(0, 0, 0, .03)',
                                  itemOpacity: 1,
                                  itemTextColor: theme === 'dark' ? '#f1f5f9' : '#1e293b'
                                }
                              }
                            ]
                          }
                        ]}
                        theme={{
                          background: 'transparent',
                          text: {
                            fontSize: 12,
                            fill: theme === 'dark' ? '#94a3b8' : '#64748b',
                            fontFamily: 'inherit'
                          },
                          axis: {
                            domain: {
                              line: {
                                stroke: theme === 'dark' ? '#334155' : '#e2e8f0',
                                strokeWidth: 1
                              }
                            },
                            legend: {
                              text: {
                                fontSize: 12,
                                fill: theme === 'dark' ? '#94a3b8' : '#64748b',
                                fontFamily: 'inherit'
                              }
                            },
                            ticks: {
                              line: {
                                stroke: theme === 'dark' ? '#334155' : '#e2e8f0',
                                strokeWidth: 1
                              },
                              text: {
                                fontSize: 11,
                                fill: theme === 'dark' ? '#94a3b8' : '#64748b',
                                fontFamily: 'inherit'
                              }
                            }
                          },
                          grid: {
                            line: {
                              stroke: theme === 'dark' ? '#334155' : '#e2e8f0',
                              strokeWidth: 1,
                              strokeDasharray: '3 3'
                            }
                          },
                          tooltip: {
                            container: {
                              background: theme === 'dark' ? '#1e293b' : '#ffffff',
                              color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
                              fontSize: '14px',
                              border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                              borderRadius: '12px',
                              padding: '12px 16px',
                              boxShadow: theme === 'dark' 
                                ? '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)'
                                : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                              minWidth: '200px'
                            }
                          }
                        }}
                        layers={['grid', 'markers', 'areas', 'lines', 'axes', 'points', 'legends', 'slices', 'crosshair']}
                        enableSlices="x"
                        sliceTooltip={({ slice }) => {
                          const sortedPoints = [...slice.points].sort((a, b) => {
                            const aId = (a as any).seriesId || (a as any).serieId || '';
                            const bId = (b as any).seriesId || (b as any).serieId || '';
                            return aId.localeCompare(bId);
                          });
                          
                          return (
                            <div
                              style={{
                                background: theme === 'dark' ? '#1e293b' : '#ffffff',
                                padding: '14px 18px',
                                border: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`,
                                borderRadius: '12px',
                                fontSize: '14px',
                                color: theme === 'dark' ? '#f1f5f9' : '#1e293b',
                                boxShadow: theme === 'dark' 
                                  ? '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)'
                                  : '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                                minWidth: '220px'
                              }}
                            >
                              <div style={{ 
                                marginBottom: '10px', 
                                fontWeight: 'bold', 
                                fontSize: '15px',
                                paddingBottom: '8px',
                                borderBottom: `1px solid ${theme === 'dark' ? '#475569' : '#e2e8f0'}`
                              }}>
                                Time: {slice.points[0].data.x}
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                {sortedPoints.map(point => {
                                  const pointId = (point as any).seriesId || (point as any).serieId || '';
                                  const pointColor = (point as any).seriesColor || (point as any).serieColor || '#94a3b8';
                                  const isGrayedOut = selectedGas !== null && selectedGas !== pointId;
                                  
                                  return (
                                    <div 
                                      key={point.id} 
                                      style={{ 
                                        display: 'flex', 
                                        alignItems: 'center', 
                                        justifyContent: 'space-between',
                                        gap: '12px',
                                        padding: '6px 8px',
                                        borderRadius: '6px',
                                        backgroundColor: isGrayedOut 
                                          ? (theme === 'dark' ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.05)')
                                          : 'transparent',
                                        opacity: isGrayedOut ? 0.6 : 1
                                      }}
                                    >
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1 }}>
                                        <div
                                          style={{
                                            width: '14px',
                                            height: '14px',
                                            borderRadius: '50%',
                                            background: pointColor,
                                            border: `2px solid ${theme === 'dark' ? '#1e293b' : '#ffffff'}`,
                                            boxShadow: `0 0 0 1px ${pointColor}40`
                                          }}
                                        />
                                        <span style={{ fontWeight: '500' }}>{pointId}</span>
                                      </div>
                                      <span style={{ 
                                        fontWeight: '600',
                                        color: pointColor,
                                        fontSize: '15px'
                                      }}>
                                        {point.data.y.toFixed(1)}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        }}
                      />
                    </div>
                  </div>

                  {/* Pollutant Gauges Grid - ONLY O3 and NO2 from API */}
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Today's AQI</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6">
                      {liveData && liveData.predictions && liveData.predictions.length > 0 && (
                        <>
                          <PollutantGauge name="O₃" value={liveData.predictions[0].O3_target || 0} maxValue={150} unit="µg/m³" color="#06b6d4" />  
                          <PollutantGauge name="NO₂" value={liveData.predictions[0].NO2_target || 0} maxValue={200} unit="µg/m³" color="#3b82f6" />
                        </>
                      )}
                    </div>
                  </div>

                  {/* Station Info - from live API data */}
                  {liveData?.live_source && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                        <div className="flex items-center gap-2 mb-4">
                          <MapPin className="w-5 h-5 text-cyan-500" />
                          <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Station Information</h3>
                        </div>
                        <div className={`space-y-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Station:</span> {liveData.live_source.station_name}</p>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Location:</span> {liveData.live_source.station_location[0].toFixed(4)}°N, {liveData.live_source.station_location[1].toFixed(4)}°E</p>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>AQI:</span> <span className="text-amber-400 font-semibold">{liveData.live_source.overall_aqi}</span></p>
                        </div>
                      </div>

                      <div className={`rounded-xl border p-4 ${theme === 'dark' ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-200'}`}>
                        <div className="flex items-center gap-2 mb-4">
                          <Clock className="w-5 h-5 text-cyan-500" />
                          <h3 className={`font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Measurement Time</h3>
                        </div>
                        <div className={`space-y-2 text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Time:</span> {liveData.live_source.measurement_time}</p>
                          <p><span className={theme === 'dark' ? 'text-slate-400' : 'text-slate-500'}>Timezone:</span> {liveData.live_source.timezone}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Other Tab Content (Health, etc.) */}
              {activeTab === 'health' && healthStatus && modelDetail && (
                <HealthStatus 
                  healthStatus={healthStatus}
                  modelDetail={modelDetail}
                />
              )}
            </>
          )}
      </div>
    </div>
  );
}