'use client';

import React, { useEffect, useState } from 'react';

/**
 * Client-only wrapper component that ensures children only render on the client side
 * This prevents SSR issues with libraries like react-leaflet
 */
export default function ClientOnlyMap({ children }: { children: React.ReactNode }) {
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    // Only set mounted after component has mounted on client
    // Use requestAnimationFrame to ensure we're definitely in the browser
    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
      // Double-check with a small delay to ensure React has fully hydrated
      const timer = setTimeout(() => {
        setHasMounted(true);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, []);

  // Don't render anything on server or during initial client render
  // Also check window/document to be extra safe
  if (!hasMounted || typeof window === 'undefined' || typeof document === 'undefined') {
    return (
      <div className="relative h-[420px] w-full flex items-center justify-center bg-slate-50 dark:bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500 mb-2"></div>
          <p className="text-sm text-slate-600 dark:text-slate-400">Loading map...</p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

