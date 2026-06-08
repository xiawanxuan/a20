import os
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import BUOY_CONFIG, DATA_DIR


class BuoyDataFetcher:
    def __init__(self, use_simulation=True):
        self.use_simulation = use_simulation
        self.buoys = BUOY_CONFIG['default_buoys']
        self.data_cache = {}
        self.last_update = None

    def fetch_buoy_list(self):
        return self.buoys

    def fetch_realtime_data(self, buoy_id=None):
        if buoy_id:
            return self._generate_buoy_profile(buoy_id)
        else:
            all_data = {}
            for buoy in self.buoys:
                all_data[buoy['id']] = self._generate_buoy_profile(buoy['id'])
            self.last_update = datetime.now()
            return all_data

    def fetch_surface_data(self, buoy_id=None):
        if buoy_id:
            return self._generate_surface_reading(buoy_id)
        else:
            all_surface = {}
            for buoy in self.buoys:
                all_surface[buoy['id']] = self._generate_surface_reading(buoy['id'])
            self.last_update = datetime.now()
            return all_surface

    def fetch_historical_data(self, buoy_id, start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='h')
        profiles = []
        buoy_info = self._get_buoy_info(buoy_id)
        for dt in dates:
            profile = self._generate_profile_at_time(buoy_id, dt, buoy_info)
            profiles.append(profile)
        return pd.concat(profiles, ignore_index=True)

    def _get_buoy_info(self, buoy_id):
        for buoy in self.buoys:
            if buoy['id'] == buoy_id:
                return buoy
        return None

    def _generate_buoy_profile(self, buoy_id):
        buoy_info = self._get_buoy_info(buoy_id)
        now = datetime.now()
        return self._generate_profile_at_time(buoy_id, now, buoy_info)

    def _generate_profile_at_time(self, buoy_id, timestamp, buoy_info):
        depths = np.concatenate([
            np.arange(0, 100, 5),
            np.arange(100, 500, 10),
            np.arange(500, 1000, 25),
            np.arange(1000, 2000, 50),
        ])

        base_temp = 15 + 10 * np.sin(2 * np.pi * (timestamp.timetuple().tm_yday / 365.0) + buoy_info['lat'] * 0.05)
        base_salinity = 34.5 + 0.5 * np.sin(2 * np.pi * (timestamp.timetuple().tm_yday / 365.0))

        temperature = []
        salinity = []
        pressure = []
        current_u = []
        current_v = []

        for d in depths:
            temp_gradient = 0
            if d < 200:
                temp_gradient = (base_temp - 8) * np.exp(-d / 80)
                temp = 8 + temp_gradient + random.gauss(0, 0.2)
            else:
                temp = 8 - (d - 200) * 0.003 + random.gauss(0, 0.05)
            temperature.append(temp)

            sal_gradient = (base_salinity - 34.0) * np.exp(-d / 150)
            sal = 34.0 + sal_gradient + random.gauss(0, 0.05)
            salinity.append(sal)

            pressure.append(d * 1.025)

            u_speed = 0.3 * np.exp(-d / 200) + random.gauss(0, 0.05)
            v_speed = 0.2 * np.exp(-d / 200) + random.gauss(0, 0.05)
            current_u.append(u_speed)
            current_v.append(v_speed)

        df = pd.DataFrame({
            'buoy_id': buoy_id,
            'buoy_name': buoy_info['name'],
            'timestamp': timestamp,
            'latitude': buoy_info['lat'],
            'longitude': buoy_info['lon'],
            'depth': depths,
            'temperature': temperature,
            'salinity': salinity,
            'pressure': pressure,
            'current_u': current_u,
            'current_v': current_v,
        })

        df['current_speed'] = np.sqrt(df['current_u'] ** 2 + df['current_v'] ** 2)
        df['current_direction'] = np.degrees(np.arctan2(df['current_v'], df['current_u'])) % 360

        return df

    def _generate_surface_reading(self, buoy_id):
        profile = self._generate_buoy_profile(buoy_id)
        surface = profile[profile['depth'] == 0].iloc[0].to_dict()

        wave_height = 1.5 + random.gauss(0, 0.5)
        if wave_height < 0:
            wave_height = 0
        surface['wave_height'] = wave_height
        surface['wave_period'] = 8 + random.gauss(0, 2)
        surface['wind_speed'] = 5 + random.gauss(0, 3)
        surface['wind_direction'] = random.uniform(0, 360)
        surface['air_temperature'] = surface['temperature'] + random.uniform(-2, 3)
        return surface

    def load_offline_data(self, filepath):
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Data file not found: {filepath}")

    def save_data(self, data, filename):
        filepath = os.path.join(DATA_DIR, filename)
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, dict):
            all_dfs = []
            for buoy_id, df in data.items():
                all_dfs.append(df)
            pd.concat(all_dfs, ignore_index=True).to_csv(filepath, index=False)
        return filepath

    def get_update_status(self):
        return {
            'last_update': self.last_update,
            'buoy_count': len(self.buoys),
            'data_points': sum(len(v) for v in self.data_cache.values()) if self.data_cache else 0
        }
