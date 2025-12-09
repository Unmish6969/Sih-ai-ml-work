'use client';

import React, { useEffect, useMemo, useState } from 'react';

// Define types locally to avoid importing from leaflet
type LatLngTuple = [number, number];
type LatLngBoundsExpression = [[number, number], [number, number]];

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

type DelhiAirMapContentProps = {
  data: SiteReading[];
  pollutant: Pollutant;
  dataset: Dataset;
  center: LatLngTuple;
  bounds: LatLngBoundsExpression;
  tileUrl: string;
  gradient: Record<string, string>;
  range: { min: number; max: number; color: string; icon: React.ReactNode };
};

export default function DelhiAirMapContent({
  data,
  pollutant,
  dataset,
  center,
  bounds,
  tileUrl,
  gradient,
  range,
}: DelhiAirMapContentProps) {
  const [isMounted, setIsMounted] = useState(false);
  const [MapComponents, setMapComponents] = useState<{
    MapContainer: any;
    TileLayer: any;
    Marker: any;
    Popup: any;
    HeatLayer: React.FC<{
      points: Array<[number, number, number]>;
      radius?: number;
      blur?: number;
      gradient?: Record<string, string>;
    }>;
    FitBounds: React.FC<{ sites: SiteReading[] }>;
  } | null>(null);

  // Dynamically import react-leaflet and leaflet only on client side
  useEffect(() => {
    // Triple-check we're on client - this should never run on server
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      console.warn('DelhiAirMapContent: Skipping SSR');
      return;
    }
    
    let isCancelled = false;
    
    // Import CSS first (non-blocking)
    Promise.all([
      import('leaflet/dist/leaflet.css').catch((err) => {
        console.warn('Failed to load leaflet CSS:', err);
        return null;
      }),
    ]).then(() => {
      if (isCancelled) return;
      
      // Then import the actual modules
      return Promise.all([
        import('react-leaflet'),
        import('leaflet'),
        import('leaflet/dist/images/marker-icon-2x.png').catch(() => null),
        import('leaflet/dist/images/marker-icon.png').catch(() => null),
        import('leaflet/dist/images/marker-shadow.png').catch(() => null),
        // Try to load leaflet.heat, but don't fail if it doesn't load
        import('leaflet.heat').catch((err) => {
          console.warn('leaflet.heat not available, heatmap will be disabled:', err);
          return null;
        }),
      ]);
    }).then(([reactLeaflet, L, markerIcon2x, markerIcon, markerShadow, leafletHeat]) => {
      if (isCancelled) return;
      
      // Final check - ensure we're still on client
      if (typeof window === 'undefined') {
        console.warn('DelhiAirMapContent: Window undefined after import');
        return;
      }
      
      if (!reactLeaflet || !L) {
        throw new Error('Failed to load react-leaflet or leaflet');
      }
      
      // Verify react-leaflet exports are valid
      if (!reactLeaflet.MapContainer || !reactLeaflet.TileLayer) {
        throw new Error('react-leaflet exports are invalid');
      }
      
      const { MapContainer, TileLayer, Marker, Popup, useMap } = reactLeaflet;
      
      // Register leaflet.heat if available
      if (leafletHeat && typeof (L.default as any).heatLayer === 'undefined') {
        try {
          // Try to register the heat plugin
          if (leafletHeat.default) {
            (L.default as any).heatLayer = leafletHeat.default;
          } else if ((leafletHeat as any).heatLayer) {
            (L.default as any).heatLayer = (leafletHeat as any).heatLayer;
          }
        } catch (e) {
          console.warn('Could not register leaflet.heat plugin:', e);
        }
      }
      
      // Fix marker icons for Vite builds
      try {
        if (markerIcon && markerIcon2x && markerShadow) {
          const DefaultIcon = L.default.icon({
            iconUrl: markerIcon.default || markerIcon,
            iconRetinaUrl: markerIcon2x.default || markerIcon2x,
            shadowUrl: markerShadow.default || markerShadow,
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41],
          });
          L.default.Marker.prototype.options.icon = DefaultIcon;
        }
      } catch (e) {
        console.warn('Failed to set marker icons (using defaults):', e);
      }

      // Create HeatLayer component
      const HeatLayerComponent: React.FC<{
        points: Array<[number, number, number]>;
        radius?: number;
        blur?: number;
        gradient?: Record<string, string>;
      }> = ({ points, radius = 28, blur = 18, gradient }) => {
        const map = useMap();

        useEffect(() => {
          if (!map) return;
          
          // Check if heatLayer is available
          if (typeof (L.default as any).heatLayer !== 'function') {
            console.warn('leaflet.heat not available, skipping heat layer');
            return;
          }
          
          try {
            const layer = (L.default as any).heatLayer(points, {
              radius,
              blur,
              maxZoom: 17,
              minOpacity: 0.15,
              gradient,
            });
            layer.addTo(map);
            return () => {
              try {
                map.removeLayer(layer);
              } catch (e) {
                // Ignore cleanup errors
              }
            };
          } catch (e) {
            console.error('Error creating heat layer:', e);
          }
        }, [map, points, radius, blur, gradient]);

        return null;
      };

      // Create FitBounds component
      const FitBoundsComponent: React.FC<{ sites: SiteReading[] }> = ({ sites }) => {
        const map = useMap();

        useEffect(() => {
          if (!map || sites.length === 0) return;
          
          try {
            const bounds = L.default.latLngBounds(
              sites.map(site => [site.lat, site.lon] as LatLngTuple)
            );
            
            map.fitBounds(bounds, {
              padding: [50, 50],
              maxZoom: 12,
            });
          } catch (e) {
            console.error('Error fitting bounds:', e);
          }
        }, [map, sites]);

        return null;
      };

      setMapComponents({
        MapContainer,
        TileLayer,
        Marker,
        Popup,
        HeatLayer: HeatLayerComponent,
        FitBounds: FitBoundsComponent,
      });
      setIsMounted(true);
      console.log('Map components loaded successfully');
    }).catch((error) => {
      console.error('Error loading map components:', error);
      // Set a flag to show error state
      setIsMounted(false);
    });
    
    return () => {
      isCancelled = true;
    };
  }, []);

  const heatPoints = useMemo(() => {
    return data.map((site) => {
      const key = `${pollutant}_${dataset}` as keyof SiteReading;
      const raw = site[key] as number;
      const clamp = (v: number) => Math.max(range.min, Math.min(range.max, v));
      const normalized = (clamp(raw) - range.min) / (range.max - range.min || 1);
      return [site.lat, site.lon, normalized] as [number, number, number];
    });
  }, [data, pollutant, dataset, range]);

  // Don't render map until components are loaded and mounted
  if (!isMounted || !MapComponents) {
    return (
      <div className="relative h-[420px] w-full flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mb-2"></div>
          <p className="text-sm text-slate-600 dark:text-slate-400">Loading map...</p>
          <p className="text-xs text-slate-500 dark:text-slate-500 mt-1">This may take a few seconds</p>
        </div>
      </div>
    );
  }

  const { MapContainer, TileLayer, Marker, Popup, HeatLayer, FitBounds } = MapComponents;

  return (
    <div className="relative h-[420px] w-full">
      <MapContainer
        {...({
          center: center as [number, number],
          zoom: 11,
          minZoom: 9,
          maxZoom: 16,
          maxBounds: bounds,
          maxBoundsViscosity: 1.0,
          scrollWheelZoom: true,
          className: 'h-full w-full',
        } as any)}
      >
        <TileLayer {...({ attribution: '&copy; OpenStreetMap', url: tileUrl } as any)} />
        <FitBounds sites={data} />
        <HeatLayer points={heatPoints} gradient={gradient} />

        {data.map((site) => {
          const key = `${pollutant}_${dataset}` as keyof SiteReading;
          const value = site[key] as number;
          return (
            <Marker key={`${pollutant}-${site.id}`} position={[site.lat, site.lon]}>
              <Popup>
                <div className="space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-slate-800">
                    <span>{site.name}</span>
                  </div>
                  <div className="text-sm text-slate-600">
                    <div className="flex items-center gap-2">
                      <span>
                        {pollutant === 'O3' ? 'O₃' : 'NO₂'} ({dataset}):{' '}
                        <span className="font-semibold text-slate-900">{value.toFixed(1)} ppb</span>
                      </span>
                    </div>
                    <div className="text-xs text-slate-500 mt-1">Updated: {new Date(site.updatedAt).toLocaleString()}</div>
                  </div>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      <div
        className={`absolute right-3 bottom-3 rounded-lg border px-3 py-2 shadow-md bg-white/90 border-slate-200 text-slate-700`}
      >
        <div className="text-xs font-semibold mb-1 flex items-center gap-1" style={{ color: range.color }}>
          {range.icon}
          <span>{pollutant === 'O3' ? 'O₃' : 'NO₂'} ({dataset})</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2 w-32 rounded-full" style={{ background: 'linear-gradient(90deg, #0ea5e9, #22d3ee, #a3e635, #f59e0b, #ef4444)' }} />
          <div className="text-[10px] text-slate-500">{range.min}–{range.max} ppb</div>
        </div>
      </div>
    </div>
  );
}






