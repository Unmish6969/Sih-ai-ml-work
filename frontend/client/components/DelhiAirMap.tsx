'use client';

import React, { lazy, Suspense, useMemo, useState } from 'react';
import { useTheme } from '@/contexts/ThemeContext';
import { RefreshCw, Flame, Droplets } from 'lucide-react';
import ClientOnlyMap from './ClientOnlyMap';

// Define types locally to avoid importing from leaflet
type LatLngTuple = [number, number];
type LatLngBoundsExpression = [[number, number], [number, number]];

// Dynamically import the map content component to avoid SSR issues
// Only import when actually needed (inside ClientOnlyMap wrapper)
// Use a function to ensure it's only called on client side
const DelhiAirMapContent = lazy(() => {
  // Double-check we're on client before importing
  if (typeof window === 'undefined') {
    // Return a no-op component for SSR
    return Promise.resolve({ 
      default: () => null as any
    } as any);
  }
  return import('./DelhiAirMapContent').then(mod => ({ default: mod.default }));
});

type Pollutant = 'O3' | 'NO2';
type Dataset = 'actual' | 'predicted';

type SiteReading = {
  id: number;
  name: string;
  lat: number;
  lon: number;
  O3_actual: number;
  O3_predicted: number;
  NO2_actual: number;
  NO2_predicted: number;
  updatedAt: string;
};

type DelhiAirMapProps = {
  sites?: SiteReading[];
  onRefresh?: () => void;
};

const defaultSites: SiteReading[] = [
  { id: 1, name: 'Site 1', lat: 28.69536, lon: 77.18168, O3_actual: 48, O3_predicted: 52, NO2_actual: 60, NO2_predicted: 58, updatedAt: new Date().toISOString() },
  { id: 2, name: 'Site 2', lat: 28.5718, lon: 77.07125, O3_actual: 42, O3_predicted: 46, NO2_actual: 55, NO2_predicted: 53, updatedAt: new Date().toISOString() },
  { id: 3, name: 'Site 3', lat: 28.58278, lon: 77.23441, O3_actual: 50, O3_predicted: 54, NO2_actual: 63, NO2_predicted: 61, updatedAt: new Date().toISOString() },
  { id: 4, name: 'Site 4', lat: 28.82286, lon: 77.10197, O3_actual: 44, O3_predicted: 48, NO2_actual: 52, NO2_predicted: 50, updatedAt: new Date().toISOString() },
  { id: 5, name: 'Site 5', lat: 28.53077, lon: 77.27123, O3_actual: 46, O3_predicted: 51, NO2_actual: 58, NO2_predicted: 57, updatedAt: new Date().toISOString() },
  { id: 6, name: 'Site 6', lat: 28.72954, lon: 77.09601, O3_actual: 49, O3_predicted: 53, NO2_actual: 57, NO2_predicted: 55, updatedAt: new Date().toISOString() },
  { id: 7, name: 'Site 7', lat: 28.71052, lon: 77.24951, O3_actual: 47, O3_predicted: 50, NO2_actual: 59, NO2_predicted: 58, updatedAt: new Date().toISOString() },
];

const pollutantRanges: Record<Pollutant, { min: number; max: number; color: string; icon: React.ReactNode }> = {
  O3: { min: 0, max: 150, color: '#06b6d4', icon: <Droplets className="w-4 h-4" /> },
  NO2: { min: 0, max: 200, color: '#f97316', icon: <Flame className="w-4 h-4" /> },
};

// Fix marker icons for Vite builds
const DefaultIcon = L.icon({
  iconUrl: markerIcon,
  iconRetinaUrl: markerIcon2x,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

const HeatLayer: React.FC<{
  points: Array<[number, number, number]>;
  radius?: number;
  blur?: number;
  gradient?: Record<string, string>;
}> = ({ points, radius = 28, blur = 18, gradient }) => {
  const map = useMap();

  useEffect(() => {
    if (!map) return;
    const layer = (L as any).heatLayer(points, {
      radius,
      blur,
      maxZoom: 17,
      minOpacity: 0.15,
      gradient,
    });
    layer.addTo(map);
    return () => {
      map.removeLayer(layer);
    };
  }, [map, points, radius, blur, gradient]);

  return null;
};

// Component to fit map bounds to show all markers
const FitBounds: React.FC<{ sites: SiteReading[] }> = ({ sites }) => {
  const map = useMap();

  useEffect(() => {
    if (!map || sites.length === 0) return;

    // Defer until map is ready and has size to avoid _leaflet_pos errors
    const doFit = () => {
      if (!map || sites.length === 0) return;
      const size = map.getSize();
      if (!size || size.x === 0 || size.y === 0) return; // skip if container not laid out yet

      const bounds = L.latLngBounds(
        sites.map(site => [site.lat, site.lon] as LatLngTuple)
      );

      map.fitBounds(bounds, {
        padding: [50, 50],
        maxZoom: 12,
      });
    };

    map.whenReady(() => {
      // small timeout to ensure layout pass completed
      setTimeout(doFit, 50);
    });

    return () => {
      // no cleanup needed
    };
  }, [map, sites]);

  return null;
};

export default function DelhiAirMap({ sites, onRefresh }: DelhiAirMapProps) {
  const { theme } = useTheme();
  const [dataset, setDataset] = useState<Dataset>('actual');

  const data = sites ?? defaultSites;
  
  // Calculate center and bounds dynamically from site locations
  // These hooks must be called unconditionally (before any early returns)
  const center: LatLngTuple = useMemo(() => {
    if (data.length === 0) return [28.61, 77.21];
    const avgLat = data.reduce((sum, site) => sum + site.lat, 0) / data.length;
    const avgLon = data.reduce((sum, site) => sum + site.lon, 0) / data.length;
    return [avgLat, avgLon];
  }, [data]);

  const bounds: LatLngBoundsExpression = useMemo(() => {
    if (data.length === 0) {
      return [
        [28.35, 76.8],
        [28.95, 77.45],
      ];
    }
    const lats = data.map(site => site.lat);
    const lons = data.map(site => site.lon);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    // Add padding to bounds
    const latPadding = (maxLat - minLat) * 0.1;
    const lonPadding = (maxLon - minLon) * 0.1;
    return [
      [minLat - latPadding, minLon - lonPadding],
      [maxLat + latPadding, maxLon + lonPadding],
    ];
  }, [data]);

  const gradient = useMemo(
    () => ({
      0.0: '#0ea5e9',
      0.25: '#22d3ee',
      0.5: '#a3e635',
      0.75: '#f59e0b',
      1.0: '#ef4444',
    }),
    []
  );

  const tileUrl = useMemo(
    () =>
      theme === 'dark'
        ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
        : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    [theme]
  );

  return (
    <ClientOnlyMap>
      <div className={`rounded-xl border ${theme === 'dark' ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200'} overflow-hidden`}>
        <div className="flex flex-col gap-4 p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className={`text-lg font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>Delhi O₃ & NO₂ Heatmaps</h3>
            <p className={`text-sm ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Side-by-side intensity for Delhi sites only</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className={`flex rounded-lg p-1 ${theme === 'dark' ? 'bg-slate-800' : 'bg-slate-100'}`}>
              {(['actual', 'predicted'] as Dataset[]).map((d) => (
                <button
                  key={d}
                  onClick={() => setDataset(d)}
                  className={`px-3 py-1.5 text-sm rounded-md font-medium transition ${
                    dataset === d
                      ? 'bg-emerald-600 text-white shadow-sm'
                      : theme === 'dark'
                      ? 'text-slate-200 hover:bg-slate-700'
                      : 'text-slate-700 hover:bg-white'
                  }`}
                >
                  {d === 'actual' ? 'Actual' : 'Predicted'}
                </button>
              ))}
            </div>
            {onRefresh && (
              <button
                onClick={onRefresh}
                className={`inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition ${
                  theme === 'dark'
                    ? 'bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700'
                    : 'bg-white text-slate-700 hover:bg-slate-100 border border-slate-200'
                }`}
              >
                <RefreshCw className="w-4 h-4" /> Refresh
              </button>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-2">
          {(['NO2', 'O3'] as Pollutant[]).map((pollutant) => {
            const range = pollutantRanges[pollutant];

            return (
              <div key={pollutant} className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {range.icon}
                    <div>
                      <p className={`text-sm font-semibold ${theme === 'dark' ? 'text-white' : 'text-slate-900'}`}>{pollutant === 'O3' ? 'O₃' : 'NO₂'} Heatmap</p>
                      <p className={`text-xs ${theme === 'dark' ? 'text-slate-300' : 'text-slate-600'}`}>Delhi focus • {dataset === 'actual' ? 'Actual readings' : 'Model predictions'}</p>
                    </div>
                  </div>
                </div>

                <Suspense fallback={
                  <div className="relative h-[420px] w-full flex items-center justify-center">
                    <p className={`text-sm ${theme === 'dark' ? 'text-slate-400' : 'text-slate-600'}`}>Loading map...</p>
                  </div>
                }>
                  <DelhiAirMapContent
                    data={data}
                    pollutant={pollutant}
                    dataset={dataset}
                    center={center}
                    bounds={bounds}
                    tileUrl={tileUrl}
                    gradient={gradient}
                    range={range}
                  />
                </Suspense>
              </div>
            );
          })}
        </div>
      </div>
    </ClientOnlyMap>
  );
}


