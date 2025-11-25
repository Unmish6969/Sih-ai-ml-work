"""
Service that fetches live air-quality observations (NO2/O3/weather) from WAQI
and converts them into the feature format expected by the ML models.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = BACKEND_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=True)


class LiveDataService:
    """Fetches live WAQI data and maps it to model-ready features."""

    BASE_URL = "https://api.waqi.info/feed"

    def __init__(
        self,
        lat_lon_path: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: int = 10,
    ):
        token = (api_token or os.getenv("WAQI_API_TOKEN") or "demo").strip()
        if token.lower() == "demo":
            raise ValueError(
                "WAQI_API_TOKEN is not configured. Please set it in Backend/.env or environment."
            )

        self.api_token = token
        self.timeout = timeout
        self.session = requests.Session()
        self.lat_lon_path = Path(lat_lon_path) if lat_lon_path else self._default_lat_lon_path()
        self.site_coordinates = self._load_site_coordinates(self.lat_lon_path)

    @staticmethod
    def _default_lat_lon_path() -> Path:
        default_file = BACKEND_ROOT / "Data_SIH_2025" / "lat_lon_sites.txt"
        if default_file.exists():
            return default_file
        raise FileNotFoundError(
            f"lat_lon_sites.txt not found at expected backend path: {default_file}. "
            "Please copy the coordinates file into Backend/Data_SIH_2025/."
        )

    @staticmethod
    def _load_site_coordinates(file_path: Path) -> Dict[int, Dict[str, float]]:
        coordinates: Dict[int, Dict[str, float]] = {}
        if not file_path.exists():
            raise FileNotFoundError(f"Site coordinates file not found: {file_path}")

        with open(file_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.lower().startswith("site"):
                    continue
                parts = re.split(r"[,\s]+", line)
                if len(parts) < 3:
                    continue
                try:
                    site_id = int(parts[0])
                    latitude = float(parts[1])
                    longitude = float(parts[2])
                except ValueError as exc:
                    raise ValueError(f"Invalid lat/lon line '{line}' in {file_path}") from exc
                coordinates[site_id] = {"latitude": latitude, "longitude": longitude}

        if not coordinates:
            raise ValueError(f"No coordinates parsed from {file_path}")
        return coordinates

    def _fetch_air_quality_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        location = f"geo:{latitude};{longitude}"
        url = f"{self.BASE_URL}/{location}/"
        params = {"token": self.api_token}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            message = data.get("message", "Unknown WAQI error")
            raise ValueError(f"WAQI API error: {message}")
        return data

    @staticmethod
    def _parse_measurement_datetime(time_info: Dict[str, Any]) -> Optional[datetime]:
        time_str = time_info.get("s") or time_info.get("iso")
        if not time_str:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(time_str)
        except ValueError:
            return None

    @staticmethod
    def _get_iaqi_value(iaqi: Dict[str, Any], key: str) -> Optional[float]:
        value_obj = iaqi.get(key)
        if isinstance(value_obj, dict):
            return value_obj.get("v")
        if isinstance(value_obj, (int, float)):
            return float(value_obj)
        return None

    def _extract_features(self, data_section: Dict[str, Any]) -> Dict[str, Any]:
        iaqi = data_section.get("iaqi", {})
        measurement_dt = self._parse_measurement_datetime(data_section.get("time", {})) or datetime.now()

        features: Dict[str, Any] = {
            "year": measurement_dt.year,
            "month": measurement_dt.month,
            "day": measurement_dt.day,
            "hour": measurement_dt.hour,
        }

        def set_feature(name: str, value: Optional[float]):
            if value is not None:
                features[name] = value

        set_feature("O3_forecast", self._get_iaqi_value(iaqi, "o3"))
        set_feature("NO2_forecast", self._get_iaqi_value(iaqi, "no2"))
        set_feature("T_forecast", self._get_iaqi_value(iaqi, "t"))
        set_feature("q_forecast", self._get_iaqi_value(iaqi, "h"))  # humidity proxy
        set_feature("w_forecast", self._get_iaqi_value(iaqi, "w"))

        return features

    def get_site_features(self, site_id: int) -> Dict[str, Any]:
        """Return feature dict plus metadata for a specific monitoring site."""
        if site_id not in self.site_coordinates:
            raise ValueError(f"Unknown site_id {site_id}. Available: {sorted(self.site_coordinates.keys())}")

        coords = self.site_coordinates[site_id]
        raw = self._fetch_air_quality_data(coords["latitude"], coords["longitude"])
        data_section = raw.get("data", {})

        features = self._extract_features(data_section)
        features["site_id"] = site_id

        iaqi = data_section.get("iaqi", {})
        station_geo = data_section.get("city", {}).get("geo")
        metadata = {
            "station_id": data_section.get("idx"),
            "station_name": data_section.get("city", {}).get("name"),
            "station_location": station_geo,
            "measurement_time": data_section.get("time", {}).get("s"),
            "timezone": data_section.get("time", {}).get("tz"),
            "overall_aqi": data_section.get("aqi"),
            "observed_no2": self._get_iaqi_value(iaqi, "no2"),
            "observed_o3": self._get_iaqi_value(iaqi, "o3"),
            "requested_coordinates": coords,
        }

        if station_geo and isinstance(station_geo, list) and len(station_geo) >= 2:
            station_lat, station_lon = station_geo[0], station_geo[1]
            if (
                abs(station_lat - coords["latitude"]) > 0.5
                or abs(station_lon - coords["longitude"]) > 0.5
            ):
                raise ValueError(
                    "WAQI returned data for a different location. "
                    f"Requested ({coords['latitude']}, {coords['longitude']}), "
                    f"but received station at ({station_lat}, {station_lon}) -> {metadata.get('station_name')}. "
                    "Please verify your WAQI API token/limits."
                )

        return {
            "site_id": site_id,
            "features": features,
            "metadata": metadata,
            "raw_response": raw,
        }

