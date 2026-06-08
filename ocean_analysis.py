import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import Akima1DInterpolator


class OceanAnalyzer:
    def __init__(self):
        self.analysis_results = {}

    def analyze_profile(self, df):
        results = {}
        results['thermocline'] = self.detect_thermocline(df)
        results['halocline'] = self.detect_halocline(df)
        results['pycnocline'] = self.detect_pycnocline(df)
        results['mixed_layer_depth'] = self.calculate_mld(df)
        results['water_masses'] = self.classify_water_masses(df)
        results['eddies'] = self.detect_eddies(df)
        self.analysis_results = results
        return results

    def detect_thermocline(self, df):
        depths = df['depth'].values
        temps = df['temperature'].values

        if len(depths) < 10:
            return {'found': False, 'reason': 'insufficient_data'}

        temps_smooth = savgol_filter(temps, min(11, len(temps) // 2 * 2 - 1), 2)
        dTdz = np.gradient(temps_smooth, depths)
        d2Tdz2 = np.gradient(dTdz, depths)

        dTdz_abs = np.abs(dTdz)

        peaks, properties = find_peaks(
            dTdz_abs,
            distance=max(3, len(depths) // 20),
            prominence=0.005
        )

        if len(peaks) == 0:
            return {'found': False, 'reason': 'no_peak_detected'}

        main_peak_idx = peaks[np.argmax(dTdz_abs[peaks])]

        thermocline_top_idx = main_peak_idx
        for i in range(main_peak_idx, 0, -1):
            if dTdz_abs[i] < dTdz_abs[main_peak_idx] * 0.2:
                thermocline_top_idx = i
                break

        thermocline_bottom_idx = main_peak_idx
        for i in range(main_peak_idx, len(depths) - 1):
            if dTdz_abs[i] < dTdz_abs[main_peak_idx] * 0.2:
                thermocline_bottom_idx = i
                break

        result = {
            'found': True,
            'depth': float(depths[main_peak_idx]),
            'top_depth': float(depths[thermocline_top_idx]),
            'bottom_depth': float(depths[thermocline_bottom_idx]),
            'thickness': float(depths[thermocline_bottom_idx] - depths[thermocline_top_idx]),
            'max_gradient': float(dTdz_abs[main_peak_idx]),
            'average_gradient': float(np.mean(dTdz_abs[thermocline_top_idx:thermocline_bottom_idx + 1])),
            'temperature_jump': float(np.abs(temps_smooth[thermocline_top_idx] - temps_smooth[thermocline_bottom_idx])),
            'strength': self._classify_strength(dTdz_abs[main_peak_idx], 'temperature'),
            'gradient_profile': dTdz.tolist(),
        }

        return result

    def detect_halocline(self, df):
        depths = df['depth'].values
        salinities = df['salinity'].values

        if len(depths) < 10:
            return {'found': False, 'reason': 'insufficient_data'}

        sal_smooth = savgol_filter(salinities, min(11, len(salinities) // 2 * 2 - 1), 2)
        dSdz = np.gradient(sal_smooth, depths)
        dSdz_abs = np.abs(dSdz)

        peaks, properties = find_peaks(
            dSdz_abs,
            distance=max(3, len(depths) // 20),
            prominence=0.002
        )

        if len(peaks) == 0:
            return {'found': False, 'reason': 'no_peak_detected'}

        main_peak_idx = peaks[np.argmax(dSdz_abs[peaks])]

        halo_top_idx = main_peak_idx
        for i in range(main_peak_idx, 0, -1):
            if dSdz_abs[i] < dSdz_abs[main_peak_idx] * 0.2:
                halo_top_idx = i
                break

        halo_bottom_idx = main_peak_idx
        for i in range(main_peak_idx, len(depths) - 1):
            if dSdz_abs[i] < dSdz_abs[main_peak_idx] * 0.2:
                halo_bottom_idx = i
                break

        result = {
            'found': True,
            'depth': float(depths[main_peak_idx]),
            'top_depth': float(depths[halo_top_idx]),
            'bottom_depth': float(depths[halo_bottom_idx]),
            'thickness': float(depths[halo_bottom_idx] - depths[halo_top_idx]),
            'max_gradient': float(dSdz_abs[main_peak_idx]),
            'average_gradient': float(np.mean(dSdz_abs[halo_top_idx:halo_bottom_idx + 1])),
            'salinity_jump': float(np.abs(sal_smooth[halo_top_idx] - sal_smooth[halo_bottom_idx])),
            'strength': self._classify_strength(dSdz_abs[main_peak_idx], 'salinity'),
            'gradient_profile': dSdz.tolist(),
        }

        return result

    def detect_pycnocline(self, df):
        depths = df['depth'].values
        densities = self._calculate_density(df['temperature'].values, df['salinity'].values)

        if len(depths) < 10:
            return {'found': False, 'reason': 'insufficient_data'}

        dens_smooth = savgol_filter(densities, min(11, len(densities) // 2 * 2 - 1), 2)
        dRhodz = np.gradient(dens_smooth, depths)

        peaks, properties = find_peaks(
            dRhodz,
            distance=max(3, len(depths) // 20),
            prominence=0.0001
        )

        if len(peaks) == 0:
            return {'found': False, 'reason': 'no_peak_detected'}

        main_peak_idx = peaks[np.argmax(dRhodz[peaks])]

        pycno_top_idx = main_peak_idx
        for i in range(main_peak_idx, 0, -1):
            if dRhodz[i] < dRhodz[main_peak_idx] * 0.2:
                pycno_top_idx = i
                break

        pycno_bottom_idx = main_peak_idx
        for i in range(main_peak_idx, len(depths) - 1):
            if dRhodz[i] < dRhodz[main_peak_idx] * 0.2:
                pycno_bottom_idx = i
                break

        dRhodz_safe = np.maximum(dRhodz, 0)
        buoyancy_freq = np.sqrt(9.81 / 1025.0 * dRhodz_safe)
        buoyancy_freq = np.where(np.isfinite(buoyancy_freq), buoyancy_freq, 0)

        result = {
            'found': True,
            'depth': float(depths[main_peak_idx]),
            'top_depth': float(depths[pycno_top_idx]),
            'bottom_depth': float(depths[pycno_bottom_idx]),
            'thickness': float(depths[pycno_bottom_idx] - depths[pycno_top_idx]),
            'max_gradient': float(dRhodz[main_peak_idx]),
            'average_gradient': float(np.mean(dRhodz[pycno_top_idx:pycno_bottom_idx + 1])),
            'density_jump': float(dens_smooth[pycno_bottom_idx] - dens_smooth[pycno_top_idx]),
            'max_buoyancy_freq': float(np.max(buoyancy_freq)),
            'strength': self._classify_strength(dRhodz[main_peak_idx], 'density'),
            'gradient_profile': dRhodz.tolist(),
            'buoyancy_frequency': buoyancy_freq.tolist(),
        }

        return result

    def calculate_mld(self, df, threshold=0.5):
        depths = df['depth'].values
        temps = df['temperature'].values

        if len(depths) < 5:
            return {'found': False, 'reason': 'insufficient_data'}

        surface_temp = temps[0]
        for i in range(len(depths)):
            if depths[i] >= 10:
                surface_temp = temps[i]
                break

        mld_idx = None
        for i in range(len(depths)):
            if abs(temps[i] - surface_temp) >= threshold:
                mld_idx = i
                break

        if mld_idx is None:
            return {'found': False, 'reason': 'mixed_layer_too_deep'}

        result = {
            'found': True,
            'depth': float(depths[mld_idx]),
            'temperature_diff': float(temps[mld_idx] - surface_temp),
            'threshold': threshold,
            'surface_temperature': float(surface_temp),
        }

        return result

    def classify_water_masses(self, df):
        depths = df['depth'].values
        temps = df['temperature'].values
        salinities = df['salinity'].values

        masses = []

        if len(depths) < 5:
            return masses

        thermo = self.detect_thermocline(df)
        mld_result = self.calculate_mld(df)

        if mld_result['found']:
            mld = mld_result['depth']
        else:
            mld = 50.0

        masses.append({
            'name': 'Surface Mixed Layer',
            'type': 'surface',
            'top_depth': 0.0,
            'bottom_depth': float(mld),
            'avg_temperature': float(np.mean(temps[depths <= mld])),
            'avg_salinity': float(np.mean(salinities[depths <= mld])),
            'description': 'Well-mixed surface layer due to wind stirring',
        })

        if thermo['found']:
            masses.append({
                'name': 'Thermocline',
                'type': 'thermocline',
                'top_depth': float(thermo['top_depth']),
                'bottom_depth': float(thermo['bottom_depth']),
                'avg_temperature': float(np.mean(temps[(depths >= thermo['top_depth']) & (depths <= thermo['bottom_depth'])])),
                'avg_salinity': float(np.mean(salinities[(depths >= thermo['top_depth']) & (depths <= thermo['bottom_depth'])])),
                'description': f'{thermo["strength"]} thermocline with strong temperature gradient',
            })

            deep_temp = np.mean(temps[depths > thermo['bottom_depth']])
            deep_sal = np.mean(salinities[depths > thermo['bottom_depth']])

            masses.append({
                'name': 'Deep Water',
                'type': 'deep',
                'top_depth': float(thermo['bottom_depth']),
                'bottom_depth': float(depths[-1]),
                'avg_temperature': float(deep_temp),
                'avg_salinity': float(deep_sal),
                'description': 'Cold, relatively uniform deep water mass',
            })
        else:
            masses.append({
                'name': 'Water Column',
                'type': 'well_mixed',
                'top_depth': 0.0,
                'bottom_depth': float(depths[-1]),
                'avg_temperature': float(np.mean(temps)),
                'avg_salinity': float(np.mean(salinities)),
                'description': 'Well-mixed water column (no strong thermocline)',
            })

        return masses

    def detect_eddies(self, df):
        depths = df['depth'].values
        temps = df['temperature'].values
        salinities = df['salinity'].values
        current_speeds = df['current_speed'].values if 'current_speed' in df.columns else None
        current_u = df['current_u'].values if 'current_u' in df.columns else None
        current_v = df['current_v'].values if 'current_v' in df.columns else None

        eddies = []

        if len(depths) < 20:
            return eddies

        temps_smooth = gaussian_filter1d(temps, sigma=2.0)
        sal_smooth = gaussian_filter1d(salinities, sigma=2.0)

        temp_anomaly = temps - temps_smooth
        sal_anomaly = salinities - sal_smooth

        combined_anomaly = (temp_anomaly / np.std(temp_anomaly) + sal_anomaly / np.std(sal_anomaly)) / 2

        abs_anomaly = np.abs(combined_anomaly)

        peaks, properties = find_peaks(
            abs_anomaly,
            distance=max(5, len(depths) // 15),
            prominence=0.3,
            width=2
        )

        if len(peaks) == 0 and current_speeds is not None:
            speed_anomaly = current_speeds - gaussian_filter1d(current_speeds, sigma=3.0)
            speed_peaks, _ = find_peaks(
                np.abs(speed_anomaly),
                distance=max(5, len(depths) // 15),
                prominence=0.1
            )
            peaks = np.concatenate([peaks, speed_peaks]) if len(peaks) > 0 else speed_peaks

        for peak_idx in peaks:
            peak_depth = depths[peak_idx]

            left_idx = peak_idx
            for i in range(peak_idx, 0, -1):
                if abs_anomaly[i] < abs_anomaly[peak_idx] * 0.3:
                    left_idx = i
                    break

            right_idx = peak_idx
            for i in range(peak_idx, len(depths) - 1):
                if abs_anomaly[i] < abs_anomaly[peak_idx] * 0.3:
                    right_idx = i
                    break

            top_depth = depths[left_idx]
            bottom_depth = depths[right_idx]
            thickness = bottom_depth - top_depth

            temp_signal = temp_anomaly[peak_idx]
            sal_signal = sal_anomaly[peak_idx]

            if temp_signal > 0 and sal_signal < 0:
                eddy_type = 'warm_core'
            elif temp_signal < 0 and sal_signal > 0:
                eddy_type = 'cold_core'
            elif temp_signal > 0:
                eddy_type = 'warm_anomaly'
            elif temp_signal < 0:
                eddy_type = 'cold_anomaly'
            else:
                eddy_type = 'salinity_anomaly'

            intensity_score = min(1.0, abs_anomaly[peak_idx] / 2.0)

            if thickness < 10 or thickness > 1000:
                continue

            if current_u is not None and current_v is not None:
                vorticity = np.gradient(current_v, depths) - np.gradient(current_u, depths)
                local_vort = vorticity[peak_idx] if peak_idx < len(vorticity) else 0
                rotational = abs(local_vort) > 0.0001
            else:
                local_vort = 0
                rotational = False

            eddy = {
                'type': eddy_type,
                'is_rotational': rotational,
                'core_depth': float(peak_depth),
                'top_depth': float(top_depth),
                'bottom_depth': float(bottom_depth),
                'thickness': float(thickness),
                'core_temperature': float(temps[peak_idx]),
                'core_salinity': float(salinities[peak_idx]),
                'temperature_anomaly': float(temp_signal),
                'salinity_anomaly': float(sal_signal),
                'intensity_score': float(intensity_score),
                'vorticity': float(local_vort),
                'strength': 'strong' if intensity_score > 0.7 else 'moderate' if intensity_score > 0.4 else 'weak',
            }

            eddies.append(eddy)

        eddies.sort(key=lambda x: x['intensity_score'], reverse=True)

        return eddies

    def detect_mesoscale_eddies_from_surface(self, buoy_data_list):
        if len(buoy_data_list) < 3:
            return []

        lats = np.array([d['latitude'] for d in buoy_data_list])
        lons = np.array([d['longitude'] for d in buoy_data_list])
        sst = np.array([d['surface_temperature'] for d in buoy_data_list])
        sss = np.array([d['surface_salinity'] for d in buoy_data_list])

        eddies = []

        for i in range(len(buoy_data_list)):
            dists = np.sqrt((lats - lats[i]) ** 2 + (lons - lons[i]) ** 2)
            neighbors = dists < 10.0

            if neighbors.sum() < 3:
                continue

            local_temp_anomaly = sst[i] - np.mean(sst[neighbors])
            local_sal_anomaly = sss[i] - np.mean(sss[neighbors])

            temp_std = np.std(sst[neighbors]) if np.std(sst[neighbors]) > 0 else 1.0
            sal_std = np.std(sss[neighbors]) if np.std(sss[neighbors]) > 0 else 1.0

            combined_score = np.sqrt((local_temp_anomaly / temp_std) ** 2 + (local_sal_anomaly / sal_std) ** 2)

            if combined_score < 0.5:
                continue

            if local_temp_anomaly > 0:
                eddy_type = 'warm_core_eddy'
            else:
                eddy_type = 'cold_core_eddy'

            eddy = {
                'type': eddy_type,
                'center_lat': float(lats[i]),
                'center_lon': float(lons[i]),
                'radius_km': float(np.median(dists[neighbors] * 111)),
                'core_sst': float(sst[i]),
                'core_sss': float(sss[i]),
                'temperature_anomaly': float(local_temp_anomaly),
                'salinity_anomaly': float(local_sal_anomaly),
                'amplitude': float(combined_score),
                'strength': 'strong' if combined_score > 1.5 else 'moderate' if combined_score > 0.8 else 'weak',
                'source_buoy': buoy_data_list[i].get('buoy_id', f'buoy_{i}'),
            }

            eddies.append(eddy)

        eddies.sort(key=lambda x: x['amplitude'], reverse=True)

        unique_eddies = []
        used_indices = set()
        for i, eddy in enumerate(eddies):
            if i in used_indices:
                continue
            unique_eddies.append(eddy)
            for j in range(i + 1, len(eddies)):
                dist = np.sqrt((eddy['center_lat'] - eddies[j]['center_lat']) ** 2 +
                               (eddy['center_lon'] - eddies[j]['center_lon']) ** 2) * 111
                if dist < eddy['radius_km'] * 0.5:
                    used_indices.add(j)

        return unique_eddies

    def _calculate_density(self, temperature, salinity):
        rho0 = 1025.0
        alpha = 0.0002
        beta = 0.0007
        T0 = 10.0
        S0 = 35.0
        return rho0 * (1 - alpha * (temperature - T0) + beta * (salinity - S0))

    def _classify_strength(self, gradient, variable):
        if variable == 'temperature':
            if gradient >= 0.1:
                return 'very_strong'
            elif gradient >= 0.05:
                return 'strong'
            elif gradient >= 0.02:
                return 'moderate'
            elif gradient >= 0.005:
                return 'weak'
            else:
                return 'very_weak'
        elif variable == 'salinity':
            if gradient >= 0.05:
                return 'very_strong'
            elif gradient >= 0.02:
                return 'strong'
            elif gradient >= 0.01:
                return 'moderate'
            elif gradient >= 0.002:
                return 'weak'
            else:
                return 'very_weak'
        elif variable == 'density':
            if gradient >= 0.01:
                return 'very_strong'
            elif gradient >= 0.005:
                return 'strong'
            elif gradient >= 0.001:
                return 'moderate'
            elif gradient >= 0.0002:
                return 'weak'
            else:
                return 'very_weak'
        return 'unknown'

    def get_analysis_summary(self, results=None):
        if results is None:
            results = self.analysis_results

        summary = []

        if 'thermocline' in results and results['thermocline'].get('found'):
            t = results['thermocline']
            summary.append(f"Thermocline: {t['depth']:.0f}m depth, {t['strength'].replace('_', ' ')} ({t['max_gradient']:.3f} C/m)")

        if 'halocline' in results and results['halocline'].get('found'):
            h = results['halocline']
            summary.append(f"Halocline: {h['depth']:.0f}m depth, {h['strength'].replace('_', ' ')} ({h['max_gradient']:.3f} PSU/m)")

        if 'mixed_layer_depth' in results and results['mixed_layer_depth'].get('found'):
            mld = results['mixed_layer_depth']
            summary.append(f"Mixed Layer Depth: {mld['depth']:.0f}m")

        if 'water_masses' in results:
            summary.append(f"Water Masses Identified: {len(results['water_masses'])}")

        return summary

    def get_analysis_dict(self, results=None):
        if results is None:
            results = self.analysis_results

        eddies_count = len(results.get('eddies', []))
        water_masses_count = len(results.get('water_masses', []))

        return {
            'thermocline': results.get('thermocline', {'found': False}),
            'halocline': results.get('halocline', {'found': False}),
            'pycnocline': results.get('pycnocline', {'found': False}),
            'mixed_layer_depth': results.get('mixed_layer_depth', {'found': False}),
            'eddies_count': eddies_count,
            'water_masses_count': water_masses_count,
            'water_masses': results.get('water_masses', []),
            'eddies': results.get('eddies', []),
        }
