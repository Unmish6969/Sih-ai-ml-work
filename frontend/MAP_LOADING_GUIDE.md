# Map Loading Guide

## How the Map Loads

The Delhi Air Map component uses a **client-side only** loading strategy to avoid SSR (Server-Side Rendering) issues with `react-leaflet`.

### Loading Flow

1. **Dashboard Component** → Renders `<DelhiAirMap />` wrapped in `Suspense` and `MapErrorBoundary`
2. **ClientOnlyMap Wrapper** → Ensures map only renders after client-side mount
3. **DelhiAirMap Component** → Wraps the map UI and handles data/state
4. **DelhiAirMapContent** → Dynamically imports `react-leaflet` and `leaflet` libraries
5. **Map Renders** → Once all dependencies are loaded

### Automatic Loading

The map **automatically loads** when:
- You navigate to the Dashboard page
- The `overview` tab is active
- The component mounts on the client side

### Loading States

You'll see different loading indicators:

1. **Initial Load**: Spinner with "Loading map..." message
2. **Component Loading**: "Loading map... This may take a few seconds"
3. **Map Ready**: Two side-by-side maps (NO₂ and O₃) with heatmaps

## Troubleshooting

### Map Not Loading?

1. **Check Browser Console**
   - Open DevTools (F12)
   - Look for errors in the Console tab
   - Common errors:
     - `Failed to load leaflet CSS` - Usually non-critical
     - `Error loading map components` - Check network tab

2. **Verify Dependencies**
   ```bash
   cd Sih-ai-ml-work-dubey/frontend
   npm list leaflet react-leaflet leaflet.heat
   ```
   Should show:
   - `leaflet@^1.9.4`
   - `react-leaflet@^5.0.0`
   - `leaflet.heat@^0.2.0`

3. **Reinstall Dependencies**
   ```bash
   npm install
   ```

4. **Check Network Tab**
   - Open DevTools → Network tab
   - Filter by "leaflet" or "map"
   - Verify CSS and JS files are loading (status 200)

5. **Clear Browser Cache**
   - Hard refresh: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
   - Or clear browser cache

### Map Shows "Loading..." Forever?

1. **Check Console for Errors**
   - Look for JavaScript errors
   - Check if `react-leaflet` is failing to import

2. **Verify Internet Connection**
   - Map tiles are loaded from external CDN
   - Check if you can access: `https://basemaps.cartocdn.com`

3. **Check Browser Compatibility**
   - Modern browsers required (Chrome, Firefox, Edge, Safari)
   - Ensure JavaScript is enabled

### Map Loads But Shows Blank/White?

1. **Check CSS Loading**
   - Leaflet CSS should be loaded automatically
   - Check Network tab for `leaflet.css`

2. **Verify Map Container Height**
   - Map needs a defined height (420px)
   - Check if parent container has proper styling

3. **Check Console for Leaflet Errors**
   - Look for warnings about missing icons
   - These are usually non-critical

## Manual Testing

To test if the map loads correctly:

1. **Open Dashboard**
   ```
   Navigate to: http://localhost:8082 (or your dev server URL)
   Click on "Dashboard" in navigation
   ```

2. **Check Map Section**
   - Scroll to "Delhi O₃ & NO₂ Heatmaps" section
   - Should see two maps side-by-side
   - Each map should show:
     - Heatmap overlay
     - Markers for each site
     - Legend in bottom-right corner

3. **Interact with Map**
   - Click markers to see popups
   - Toggle between "Actual" and "Predicted" data
   - Zoom and pan the map

## Expected Behavior

✅ **Working Map:**
- Shows two maps (NO₂ and O₃)
- Heatmap colors change based on pollutant values
- Markers are clickable with popups
- Toggle between Actual/Predicted works
- Map is interactive (zoom, pan)

❌ **Not Working:**
- Stuck on "Loading map..." for > 10 seconds
- Blank/white area where map should be
- Console errors about react-leaflet
- Map tiles not loading

## Dependencies

Required packages (already in `package.json`):
- `leaflet@^1.9.4` - Core mapping library
- `react-leaflet@^5.0.0` - React bindings for Leaflet
- `leaflet.heat@^0.2.0` - Heatmap plugin

## File Structure

```
frontend/
├── client/
│   ├── components/
│   │   ├── DelhiAirMap.tsx          # Main map component
│   │   ├── DelhiAirMapContent.tsx   # Map rendering logic
│   │   └── ClientOnlyMap.tsx        # SSR prevention wrapper
│   └── main.tsx                     # Entry point
└── package.json                     # Dependencies
```

## Need Help?

If the map still doesn't load:
1. Check browser console for specific errors
2. Verify all dependencies are installed
3. Try a different browser
4. Check if dev server is running correctly

