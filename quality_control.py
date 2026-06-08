import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from config import QUALITY_CONTROL


class QualityController:
    def __init__(self, assimilation_method='optimal'):
        self.qc_rules = QUALITY_CONTROL
        self.quality_flags = {}
        self.assimilation_method = assimilation_method
        self.assimilation_stats = {}

    def run_full_qc(self, data):
        if isinstance(data, dict):
            results = {}
            for buoy_id, df in data.items():
                results[buoy_id] = self._apply_qc(df)
            return results
        else:
            return self._apply_qc(data)

    def _apply_qc(self, df):
        df = df.copy()
        df['quality_flag'] = 'good'
        df['qc_details'] = ''

        df = self._range_check(df)
        df = self._iqr_outlier_detection(df)
        df = self._gradient_check(df)
        df = self._missing_value_check(df)
        df = self._assimilate_data(df)

        return df

    def _range_check(self, df):
        rules = self.qc_rules

        temp_mask = (df['temperature'] < rules['temperature_range'][0]) | \
                    (df['temperature'] > rules['temperature_range'][1])
        df.loc[temp_mask, 'quality_flag'] = 'bad'
        df.loc[temp_mask, 'qc_details'] += 'temperature_out_of_range;'

        sal_mask = (df['salinity'] < rules['salinity_range'][0]) | \
                   (df['salinity'] > rules['salinity_range'][1])
        df.loc[sal_mask, 'quality_flag'] = 'bad'
        df.loc[sal_mask, 'qc_details'] += 'salinity_out_of_range;'

        pres_mask = (df['pressure'] < rules['pressure_range'][0]) | \
                    (df['pressure'] > rules['pressure_range'][1])
        df.loc[pres_mask, 'quality_flag'] = 'bad'
        df.loc[pres_mask, 'qc_details'] += 'pressure_out_of_range;'

        speed_mask = df['current_speed'] > rules['current_speed_max']
        df.loc[speed_mask, 'quality_flag'] = 'bad'
        df.loc[speed_mask, 'qc_details'] += 'current_speed_exceeded;'

        return df

    def _iqr_outlier_detection(self, df):
        iqr_factor = self.qc_rules['iqr_factor']
        columns = ['temperature', 'salinity', 'current_speed']

        if 'depth' in df.columns and len(df['depth'].unique()) > 1:
            for depth in df['depth'].unique():
                depth_mask = df['depth'] == depth
                for col in columns:
                    subset = df.loc[depth_mask, col]
                    q1 = subset.quantile(0.25)
                    q3 = subset.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - iqr_factor * iqr
                    upper = q3 + iqr_factor * iqr
                    outlier_mask = depth_mask & ((df[col] < lower) | (df[col] > upper))
                    suspect_mask = outlier_mask & (df['quality_flag'] == 'good')
                    df.loc[suspect_mask, 'quality_flag'] = 'suspect'
                    df.loc[suspect_mask, 'qc_details'] += f'{col}_iqr_outlier;'
        else:
            for col in columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                outlier_mask = (df[col] < lower) | (df[col] > upper)
                suspect_mask = outlier_mask & (df['quality_flag'] == 'good')
                df.loc[suspect_mask, 'quality_flag'] = 'suspect'
                df.loc[suspect_mask, 'qc_details'] += f'{col}_iqr_outlier;'

        return df

    def _gradient_check(self, df):
        if 'depth' not in df.columns or len(df) < 3:
            return df

        df_sorted = df.sort_values('depth').reset_index(drop=True)

        for col in ['temperature', 'salinity']:
            gradients = df_sorted[col].diff().abs()
            median_grad = gradients.median()
            if median_grad == 0:
                continue
            gradient_mask = gradients > 5 * median_grad
            bad_idx = df_sorted[gradient_mask].index
            for idx in bad_idx:
                if df.loc[idx, 'quality_flag'] == 'good':
                    df.loc[idx, 'quality_flag'] = 'suspect'
                    df.loc[idx, 'qc_details'] += f'{col}_abnormal_gradient;'

        return df

    def _missing_value_check(self, df):
        essential_cols = ['temperature', 'salinity', 'pressure', 'depth']
        for col in essential_cols:
            missing_mask = df[col].isna()
            df.loc[missing_mask, 'quality_flag'] = 'missing'
            df.loc[missing_mask, 'qc_details'] += f'{col}_missing;'
        return df

    def _assimilate_data(self, df):
        df = df.copy()
        df_sorted = df.sort_values('depth').reset_index(drop=True)

        temp_col = 'temperature_assimilated'
        sal_col = 'salinity_assimilated'
        speed_col = 'current_speed_assimilated'

        df[temp_col] = df['temperature'].copy()
        df[sal_col] = df['salinity'].copy()
        df[speed_col] = df['current_speed'].copy()
        df['assimilation_method'] = 'original'
        df['assimilation_confidence'] = 1.0
        df['assimilation_error'] = 0.0

        bad_mask = df_sorted['quality_flag'].isin(['bad', 'missing', 'suspect'])

        if bad_mask.any() and 'depth' in df.columns:
            depths = df_sorted['depth'].values
            good_mask = ~bad_mask

            temp_good = df_sorted.loc[good_mask, 'temperature'].values
            temp_bad_depths = depths[bad_mask]

            sal_good = df_sorted.loc[good_mask, 'salinity'].values
            sal_bad_depths = depths[bad_mask]

            speed_good = df_sorted.loc[good_mask, 'current_speed'].values

            good_depths = depths[good_mask]

            if self.assimilation_method == 'linear':
                temp_assim = self._linear_interpolation(good_depths, temp_good, temp_bad_depths)
                sal_assim = self._linear_interpolation(good_depths, sal_good, sal_bad_depths)
                speed_assim = self._linear_interpolation(good_depths, speed_good, depths[bad_mask])
                temp_err = np.full(len(temp_bad_depths), 0.15)
                sal_err = np.full(len(sal_bad_depths), 0.08)
                speed_err = np.full(len(depths[bad_mask]), 0.08)

            elif self.assimilation_method == 'spline':
                temp_assim = self._spline_interpolation(good_depths, temp_good, temp_bad_depths)
                sal_assim = self._spline_interpolation(good_depths, sal_good, sal_bad_depths)
                speed_assim = self._spline_interpolation(good_depths, speed_good, depths[bad_mask])
                temp_err = np.full(len(temp_bad_depths), 0.12)
                sal_err = np.full(len(sal_bad_depths), 0.06)
                speed_err = np.full(len(depths[bad_mask]), 0.06)

            else:
                temp_linear = self._linear_interpolation(good_depths, temp_good, temp_bad_depths)
                sal_linear = self._linear_interpolation(good_depths, sal_good, sal_bad_depths)

                temp_assim, temp_err = self._advanced_assimilation(
                    good_depths, temp_good, temp_bad_depths, depths, 'temperature'
                )
                sal_assim, sal_err = self._advanced_assimilation(
                    good_depths, sal_good, sal_bad_depths, depths, 'salinity'
                )
                speed_assim, speed_err = self._advanced_assimilation(
                    good_depths, speed_good, depths[bad_mask], depths, 'current_speed'
                )

                if self.assimilation_method in ['optimal', 'gaussian']:
                    temp_assim, sal_assim = self._apply_ts_coupling_safe(
                        temp_assim, sal_assim, temp_bad_depths,
                        good_depths, temp_good, sal_good,
                        temp_linear, sal_linear
                    )

                temp_assim = self._ensure_safety(temp_assim, temp_linear, good_depths, depths, 'temperature')
                sal_assim = self._ensure_safety(sal_assim, sal_linear, good_depths, depths, 'salinity')

            temp_assim = self._apply_ocean_constraints(
                temp_assim, temp_bad_depths, depths,
                self._estimate_background(good_depths, temp_good, depths, 'temperature'),
                'temperature'
            )
            sal_assim = self._apply_ocean_constraints(
                sal_assim, sal_bad_depths, depths,
                self._estimate_background(good_depths, sal_good, depths, 'salinity'),
                'salinity'
            )
            speed_assim = self._apply_physical_constraints(
                speed_assim, depths[bad_mask], 'current_speed'
            )

            df_sorted.loc[bad_mask, temp_col] = temp_assim
            df_sorted.loc[bad_mask, sal_col] = sal_assim
            df_sorted.loc[bad_mask, speed_col] = speed_assim
            df_sorted.loc[bad_mask, 'assimilation_method'] = self.assimilation_method
            df_sorted.loc[bad_mask, 'assimilation_confidence'] = self._compute_confidence(
                depths, good_mask, bad_mask
            )
            df_sorted.loc[bad_mask, 'assimilation_error'] = temp_err

            df = df_sorted.sort_index().reset_index(drop=True)

        df['data_assimilated'] = df['quality_flag'].isin(['bad', 'missing', 'suspect'])

        self.assimilation_stats = self._compute_assimilation_stats(df)

        return df

    def _advanced_assimilation(self, good_depths, good_values, interp_depths, all_depths, variable):
        try:
            from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
            from scipy.ndimage import gaussian_filter1d

            linear_vals = self._linear_interpolation(good_depths, good_values, interp_depths)

            try:
                akima = Akima1DInterpolator(good_depths, good_values)
                akima_vals = akima(interp_depths)
                akima_vals = np.where(np.isfinite(akima_vals), akima_vals, linear_vals)
            except Exception:
                akima_vals = linear_vals.copy()

            try:
                pchip = PchipInterpolator(good_depths, good_values)
                pchip_vals = pchip(interp_depths)
                pchip_vals = np.where(np.isfinite(pchip_vals), pchip_vals, linear_vals)
            except Exception:
                pchip_vals = linear_vals.copy()

            background = self._estimate_background(good_depths, good_values, all_depths, variable)
            bg_interp = np.interp(interp_depths, all_depths, background)

            if self.assimilation_method == 'optimal':
                w_akima = 0.70
                w_bg = 0.20
                w_linear = 0.10

                combined = w_akima * akima_vals + w_bg * bg_interp + w_linear * linear_vals

                if len(combined) >= 5:
                    try:
                        sigma = max(0.6, len(combined) / 40)
                        smoothed = gaussian_filter1d(combined, sigma=sigma)
                        combined = 0.1 * smoothed + 0.9 * combined
                    except Exception:
                        pass

            elif self.assimilation_method == 'gaussian':
                combined = 0.30 * akima_vals + 0.20 * pchip_vals + 0.30 * bg_interp + 0.20 * linear_vals

                if len(combined) >= 5:
                    try:
                        sigma = max(1.2, len(combined) / 20)
                        smoothed = gaussian_filter1d(combined, sigma=sigma)
                        combined = 0.4 * smoothed + 0.6 * combined
                    except Exception:
                        pass
            else:
                combined = 0.5 * akima_vals + 0.2 * bg_interp + 0.3 * linear_vals

            combined = np.where(np.isfinite(combined), combined, linear_vals)

            if variable == 'temperature':
                error = 0.085
            elif variable == 'salinity':
                error = 0.045
            else:
                error = 0.06

            errors = np.full(len(interp_depths), error)

            return combined, errors
        except Exception:
            result = self._spline_interpolation(good_depths, good_values, interp_depths)
            return result, np.full(len(interp_depths), 0.12)

    def _estimate_background(self, good_depths, good_values, all_depths, variable):
        try:
            from scipy.signal import savgol_filter

            if len(good_values) < 5:
                bg_good = good_values.copy()
            else:
                window = min(11, len(good_values) // 2 * 2 - 1)
                if window < 5:
                    window = 5
                if window % 2 == 0:
                    window += 1
                polyorder = min(2, window - 1)

                try:
                    bg_good = savgol_filter(good_values, window, polyorder)
                except Exception:
                    from scipy.ndimage import gaussian_filter1d
                    bg_good = gaussian_filter1d(good_values, sigma=2.0)

            f = interpolate.interp1d(good_depths, bg_good, kind='linear', fill_value='extrapolate')
            background = f(all_depths)

            return background
        except Exception:
            f = interpolate.interp1d(good_depths, good_values, kind='linear', fill_value='extrapolate')
            return f(all_depths)

    def _optimal_interpolation_full(self, good_depths, good_values, interp_depths, background_good, variable):
        if len(good_depths) < 5:
            result = self._linear_interpolation(good_depths, good_values, interp_depths)
            return result, np.full(len(interp_depths), 0.2)

        try:
            if variable == 'temperature':
                sigma_obs = 0.05
                sigma_bg = 0.3
                decorr_length = 150.0
            elif variable == 'salinity':
                sigma_obs = 0.02
                sigma_bg = 0.15
                decorr_length = 200.0
            else:
                sigma_obs = 0.03
                sigma_bg = 0.1
                decorr_length = 100.0

            def cov_func(d):
                return sigma_bg**2 * np.exp(-d / decorr_length)

            n_obs = len(good_depths)
            H = np.zeros((n_obs, n_obs))
            for i in range(n_obs):
                for j in range(n_obs):
                    d = abs(good_depths[i] - good_depths[j])
                    H[i, j] = cov_func(d)

            R = np.eye(n_obs) * sigma_obs**2
            P = H + R

            try:
                P_inv = np.linalg.inv(P)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P)

            n_interp = len(interp_depths)
            interpolated = np.zeros(n_interp)
            errors = np.zeros(n_interp)

            for i in range(n_interp):
                bd = interp_depths[i]
                h = np.array([cov_func(abs(bd - gd)) for gd in good_depths])
                innovation = good_values - background_good
                K = h @ P_inv
                bg_val = np.interp(bd, good_depths, background_good)
                interpolated[i] = bg_val + K @ innovation
                errors[i] = np.sqrt(sigma_bg**2 - K @ h)

            from scipy.ndimage import gaussian_filter1d
            if len(interpolated) >= 5:
                interpolated_smooth = gaussian_filter1d(interpolated, sigma=0.5)
                alpha = 0.7
                interpolated = alpha * interpolated_smooth + (1 - alpha) * interpolated

            return interpolated, errors
        except Exception:
            result = self._spline_interpolation(good_depths, good_values, interp_depths)
            return result, np.full(len(interp_depths), 0.1)

    def _apply_ocean_constraints(self, values, interp_depths, all_depths, background, variable):
        rules = self.qc_rules

        if variable == 'temperature':
            min_val, max_val = rules['temperature_range']
        elif variable == 'salinity':
            min_val, max_val = rules['salinity_range']
        else:
            return values

        values = np.clip(values, min_val, max_val)

        max_deviation = 2.0 if variable == 'temperature' else 1.0
        bg_interp = np.interp(interp_depths, all_depths, background)
        deviation = values - bg_interp
        too_far = np.abs(deviation) > max_deviation
        values[too_far] = bg_interp[too_far] + np.sign(deviation[too_far]) * max_deviation

        return values

    def _apply_ts_coupling(self, temp_vals, sal_vals, interp_depths, good_depths, good_temp, good_sal):
        if len(good_temp) < 8:
            return temp_vals, sal_vals

        try:
            from scipy.optimize import curve_fit

            def ts_linear(t, a, b):
                return a + b * t

            popt, _ = curve_fit(ts_linear, good_temp, good_sal, maxfev=10000)

            predicted_sal = ts_linear(temp_vals, *popt)

            alpha = 0.6
            sal_vals_corrected = alpha * predicted_sal + (1 - alpha) * sal_vals

            def st_linear(s, a, b):
                return (s - a) / b if b != 0 else s

            predicted_temp = st_linear(sal_vals_corrected, *popt)
            temp_vals_corrected = 0.3 * predicted_temp + 0.7 * temp_vals

            return temp_vals_corrected, sal_vals_corrected
        except Exception:
            return temp_vals, sal_vals

    def _enforce_static_stability(self, temp_vals, sal_vals, interp_depths, all_depths):
        if len(interp_depths) < 3:
            return temp_vals, sal_vals

        try:
            all_temp = np.interp(all_depths, interp_depths, temp_vals) if len(all_depths) != len(interp_depths) else temp_vals

            density = self._approx_density(all_temp, np.interp(all_depths, all_depths, all_depths * 0 + 34.5))

            density_sorted = np.sort(density)
            if not np.array_equal(density, density_sorted):
                alpha = 0.5
                temp_smoothed = temp_vals.copy()
                for i in range(1, len(temp_vals) - 1):
                    if temp_vals[i] < temp_vals[i+1]:
                        temp_smoothed[i] = alpha * temp_vals[i] + (1 - alpha) * (temp_vals[i-1] + temp_vals[i+1]) / 2
                temp_vals = temp_smoothed

            return temp_vals, sal_vals
        except Exception:
            return temp_vals, sal_vals

    def _enforce_static_stability_v2(self, temp_vals, sal_vals, interp_depths, all_depths,
                                     good_depths, good_temp, good_sal):
        if len(interp_depths) < 3:
            return temp_vals, sal_vals

        try:
            full_temp = np.interp(all_depths, good_depths, good_temp)
            full_sal = np.interp(all_depths, good_depths, good_sal)

            for i, depth in enumerate(interp_depths):
                idx = np.argmin(np.abs(all_depths - depth))
                full_temp[idx] = temp_vals[i]
                full_sal[idx] = sal_vals[i]

            density = self._approx_density(full_temp, full_sal)

            n = len(density)
            stable_count = 0
            max_iterations = 5

            for iteration in range(max_iterations):
                unstable = False
                for i in range(n - 1):
                    if density[i] > density[i + 1]:
                        rho_diff = density[i + 1] - density[i]
                        alpha = 0.3

                        dT_drho = -1.0 / (1025.0 * 0.0002)
                        temp_correction = rho_diff * dT_drho * alpha

                        depth_i = all_depths[i]
                        depth_ip1 = all_depths[i + 1]

                        if depth_i in interp_depths:
                            idx_interp = np.where(interp_depths == depth_i)[0][0]
                            temp_vals[idx_interp] += temp_correction * 0.5
                            full_temp[i] += temp_correction * 0.5

                        if depth_ip1 in interp_depths:
                            idx_interp = np.where(interp_depths == depth_ip1)[0][0]
                            temp_vals[idx_interp] -= temp_correction * 0.5
                            full_temp[i + 1] -= temp_correction * 0.5

                        unstable = True

                density = self._approx_density(full_temp, full_sal)

                if not unstable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            temp_vals = np.clip(temp_vals, -2.0, 40.0)
            sal_vals = np.clip(sal_vals, 28.0, 40.0)

            return temp_vals, sal_vals
        except Exception:
            return temp_vals, sal_vals

    def _approx_density(self, temp, sal):
        rho0 = 1025.0
        alpha = 0.0002
        beta = 0.0007
        return rho0 * (1 - alpha * (temp - 10) + beta * (sal - 35))

    def _ensure_safety(self, values, linear_vals, good_depths, all_depths, variable):
        try:
            n_good = len(good_depths)
            n_total = len(all_depths)
            data_density = n_good / n_total

            if data_density > 0.75:
                max_dev = 0.6
            elif data_density > 0.55:
                max_dev = 0.9
            elif data_density > 0.35:
                max_dev = 1.2
            else:
                max_dev = 1.5

            if variable == 'salinity':
                max_dev = max_dev * 0.35
            elif variable == 'current_speed':
                max_dev = max_dev * 0.5

            deviation = values - linear_vals
            abs_dev = np.abs(deviation)

            too_far = abs_dev > max_dev
            if too_far.any():
                blend_factor = np.where(too_far,
                                        max_dev / np.maximum(abs_dev, 1e-6),
                                        1.0)
                values_safe = linear_vals + blend_factor * deviation
                return values_safe

            return values
        except Exception:
            return linear_vals

    def _apply_ts_coupling_safe(self, temp_vals, sal_vals, interp_depths,
                                good_depths, good_temp, good_sal,
                                temp_linear, sal_linear):
        if len(good_temp) < 10:
            return temp_vals, sal_vals

        try:
            from scipy.optimize import curve_fit

            def ts_linear(t, a, b):
                return a + b * t

            popt, _ = curve_fit(ts_linear, good_temp, good_sal, maxfev=10000)

            predicted_sal = ts_linear(temp_vals, *popt)

            alpha_sal = 0.30
            sal_corrected = alpha_sal * predicted_sal + (1 - alpha_sal) * sal_vals

            temp_corrected = temp_vals

            max_sal_dev = 0.25
            sal_dev = np.abs(sal_corrected - sal_linear)
            too_far_sal = sal_dev > max_sal_dev
            if too_far_sal.any():
                blend = np.where(too_far_sal, max_sal_dev / np.maximum(sal_dev, 1e-6), 1.0)
                sal_corrected = sal_linear + blend * (sal_corrected - sal_linear)

            return temp_corrected, sal_corrected
        except Exception:
            return temp_vals, sal_vals

    def _enforce_static_stability_safe(self, temp_vals, sal_vals, interp_depths, all_depths,
                                       good_depths, good_temp, good_sal, temp_linear):
        if len(interp_depths) < 5:
            return temp_vals, sal_vals

        try:
            full_temp = np.interp(all_depths, good_depths, good_temp)
            full_sal = np.full(len(all_depths), 34.5)

            for i, depth in enumerate(interp_depths):
                idx = np.argmin(np.abs(all_depths - depth))
                full_temp[idx] = temp_vals[i]

            density = self._approx_density(full_temp, full_sal)

            n = len(density)
            max_iter = 10
            alpha = 0.2

            for _ in range(max_iter):
                has_unstable = False
                for i in range(n - 1):
                    if density[i] > density[i + 1]:
                        has_unstable = True
                        temp_avg = (full_temp[i] + full_temp[i + 1]) / 2
                        full_temp[i] = full_temp[i] - alpha * (full_temp[i] - temp_avg)
                        full_temp[i + 1] = full_temp[i + 1] + alpha * (temp_avg - full_temp[i + 1])

                density = self._approx_density(full_temp, full_sal)

                if not has_unstable:
                    break

            temp_result = temp_vals.copy()
            for i, depth in enumerate(interp_depths):
                idx = np.argmin(np.abs(all_depths - depth))
                temp_result[i] = full_temp[idx]

            max_deviation = 0.4
            deviation = np.abs(temp_result - temp_linear)
            too_far = deviation > max_deviation
            blend = np.where(too_far,
                             max_deviation / np.maximum(deviation, 1e-6),
                             1.0)
            temp_safe = temp_linear + blend * (temp_result - temp_linear)

            return temp_safe, sal_vals
        except Exception:
            return temp_vals, sal_vals

    def _apply_physical_constraints(self, values, depths, variable):
        rules = self.qc_rules

        if variable == 'temperature':
            min_val, max_val = rules['temperature_range']
            values = np.clip(values, min_val, max_val)

        elif variable == 'salinity':
            min_val, max_val = rules['salinity_range']
            values = np.clip(values, min_val, max_val)

        elif variable == 'current_speed':
            max_speed = rules['current_speed_max']
            values = np.clip(values, 0.0, max_speed)

        return values

    def _interpolate_variable(self, good_depths, good_values, bad_depths, variable):
        if len(good_depths) < 4:
            return self._linear_interpolation(good_depths, good_values, bad_depths)

        method = self.assimilation_method

        if method == 'spline':
            return self._spline_interpolation(good_depths, good_values, bad_depths)
        elif method == 'optimal':
            return self._optimal_interpolation(good_depths, good_values, bad_depths, variable)
        elif method == 'gaussian':
            return self._gaussian_process_interpolation(good_depths, good_values, bad_depths)
        else:
            return self._spline_interpolation(good_depths, good_values, bad_depths)

    def _linear_interpolation(self, good_depths, good_values, bad_depths):
        if len(good_depths) < 2:
            return np.full(len(bad_depths), np.mean(good_values))
        f = interpolate.interp1d(good_depths, good_values, kind='linear',
                                 fill_value='extrapolate')
        return f(bad_depths)

    def _spline_interpolation(self, good_depths, good_values, bad_depths):
        try:
            from scipy.interpolate import Akima1DInterpolator

            if len(good_depths) < 5:
                return self._linear_interpolation(good_depths, good_values, bad_depths)

            akima = Akima1DInterpolator(good_depths, good_values)
            result = akima(bad_depths)

            if np.any(np.isnan(result)):
                result = np.where(np.isnan(result),
                                  self._linear_interpolation(good_depths, good_values, bad_depths),
                                  result)

            return result
        except Exception:
            try:
                k = min(3, len(good_depths) - 1)
                tck = interpolate.splrep(good_depths, good_values, k=k, s=1.0)
                return interpolate.splev(bad_depths, tck)
            except Exception:
                return self._linear_interpolation(good_depths, good_values, bad_depths)

    def _optimal_interpolation(self, good_depths, good_values, bad_depths, variable):
        if len(good_depths) < 10:
            return self._spline_interpolation(good_depths, good_values, bad_depths)

        try:
            from scipy.signal import savgol_filter
            from scipy.ndimage import gaussian_filter1d

            window = min(15, len(good_values) // 2 * 2 - 1)
            if window < 5:
                window = 5
            if window % 2 == 0:
                window += 1
            polyorder = min(3, window - 1)

            try:
                background = savgol_filter(good_values, window, polyorder)
            except Exception:
                background = gaussian_filter1d(good_values, sigma=2.0)

            L = 120.0
            sigma_o = 0.05
            sigma_b = 0.4

            def cov_func(d):
                return sigma_b**2 * np.exp(-np.abs(d) / L)

            n_obs = len(good_depths)
            H = np.zeros((n_obs, n_obs))
            for i in range(n_obs):
                for j in range(n_obs):
                    d = abs(good_depths[i] - good_depths[j])
                    H[i, j] = cov_func(d)

            R = np.eye(n_obs) * sigma_o**2
            P = H + R

            try:
                P_inv = np.linalg.inv(P)
            except np.linalg.LinAlgError:
                P_inv = np.linalg.pinv(P)

            interpolated = np.zeros(len(bad_depths))
            for i, bd in enumerate(bad_depths):
                h = np.array([cov_func(abs(bd - gd)) for gd in good_depths])
                innovation = good_values - background
                K = h @ P_inv
                background_val = np.interp(bd, good_depths, background)
                interpolated[i] = background_val + K @ innovation

            spline_vals = self._spline_interpolation(good_depths, good_values, bad_depths)
            alpha = 0.7
            final_vals = alpha * interpolated + (1 - alpha) * spline_vals

            return final_vals
        except Exception as e:
            return self._spline_interpolation(good_depths, good_values, bad_depths)

    def _gaussian_process_interpolation(self, good_depths, good_values, bad_depths):
        try:
            from scipy.interpolate import RBFInterpolator

            rbf = RBFInterpolator(
                good_depths.reshape(-1, 1), good_values,
                kernel='inverse_quadric',
                smoothing=0.01,
                epsilon=0.02,
            )
            result = rbf(bad_depths.reshape(-1, 1))

            if np.any(np.isnan(result)) or np.any(np.abs(result) > 1e3):
                return self._spline_interpolation(good_depths, good_values, bad_depths)

            return result
        except Exception:
            return self._spline_interpolation(good_depths, good_values, bad_depths)

    def _compute_background_profile(self, depths, values, variable):
        try:
            smoothed = gaussian_filter1d(values, sigma=3)
            return smoothed
        except Exception:
            return values

    def _apply_ts_constraint(self, df, good_mask, bad_mask):
        if not bad_mask.any():
            return df

        try:
            good_temp = df.loc[good_mask, 'temperature_assimilated'].values
            good_sal = df.loc[good_mask, 'salinity_assimilated'].values
            good_depth = df.loc[good_mask, 'depth'].values

            if len(good_temp) < 5:
                return df

            from scipy.optimize import curve_fit

            def ts_relation(temp, a, b, c):
                return a + b * temp + c * temp**2

            popt, _ = curve_fit(ts_relation, good_temp, good_sal, maxfev=10000)

            bad_temp = df.loc[bad_mask, 'temperature_assimilated'].values
            predicted_sal = ts_relation(bad_temp, *popt)

            current_sal = df.loc[bad_mask, 'salinity_assimilated'].values
            alpha = 0.7
            df.loc[bad_mask, 'salinity_assimilated'] = alpha * predicted_sal + (1 - alpha) * current_sal

            return df
        except Exception:
            return df

    def _compute_confidence(self, depths, good_mask, bad_mask):
        confidences = np.zeros(bad_mask.sum())
        good_depths = depths[good_mask]

        bad_indices = np.where(bad_mask)[0]
        for i, idx in enumerate(bad_indices):
            bd = depths[idx]
            dists = np.abs(good_depths - bd)
            min_dist = np.min(dists)
            n_nearby = np.sum(dists < 100)
            conf = min(1.0, max(0.2, 1.0 - min_dist / 200.0))
            conf = conf * min(1.0, n_nearby / 3.0)
            confidences[i] = conf

        return confidences

    def _compute_assimilation_stats(self, df):
        stats = {}

        if 'data_assimilated' in df.columns and df['data_assimilated'].any():
            for var in ['temperature', 'salinity', 'current_speed']:
                assim_col = f'{var}_assimilated'
                orig_col = var
                if assim_col in df.columns:
                    good_mask = ~df['data_assimilated']
                    if good_mask.sum() > 0 and good_mask.sum() < len(df):
                        orig_good = df.loc[good_mask, orig_col].values
                        assim_good = df.loc[good_mask, assim_col].values
                        rmse = np.sqrt(np.mean((orig_good - assim_good)**2))
                        bias = np.mean(assim_good - orig_good)
                        stats[f'{var}_rmse'] = rmse
                        stats[f'{var}_bias'] = bias

        if 'assimilation_confidence' in df.columns:
            stats['mean_confidence'] = df['assimilation_confidence'].mean()

        return stats

    def get_assimilation_stats(self, data=None):
        if data is not None:
            if isinstance(data, dict):
                stats = {}
                for buoy_id, df in data.items():
                    stats[buoy_id] = self._compute_assimilation_stats(df)
                return stats
            else:
                return self._compute_assimilation_stats(data)
        return self.assimilation_stats

    def get_quality_summary(self, data):
        if isinstance(data, dict):
            summaries = {}
            for buoy_id, df in data.items():
                summaries[buoy_id] = self._compute_summary(df)
            return summaries
        else:
            return self._compute_summary(data)

    def _compute_summary(self, df):
        total = len(df)
        good = (df['quality_flag'] == 'good').sum()
        suspect = (df['quality_flag'] == 'suspect').sum()
        bad = (df['quality_flag'] == 'bad').sum()
        missing = (df['quality_flag'] == 'missing').sum()
        assimilated = df['data_assimilated'].sum() if 'data_assimilated' in df.columns else 0

        return {
            'total_points': total,
            'good_count': good,
            'suspect_count': suspect,
            'bad_count': bad,
            'missing_count': missing,
            'assimilated_count': assimilated,
            'good_percentage': round(good / total * 100, 2) if total > 0 else 0,
            'suspect_percentage': round(suspect / total * 100, 2) if total > 0 else 0,
            'bad_percentage': round(bad / total * 100, 2) if total > 0 else 0,
            'assimilation_stats': self.assimilation_stats if self.assimilation_stats else {},
        }

    def get_anomalies(self, data):
        if isinstance(data, dict):
            all_anomalies = {}
            for buoy_id, df in data.items():
                anomalies = df[df['quality_flag'].isin(['bad', 'suspect', 'missing'])]
                if len(anomalies) > 0:
                    all_anomalies[buoy_id] = anomalies
            return all_anomalies
        else:
            return data[data['quality_flag'].isin(['bad', 'suspect', 'missing'])]


