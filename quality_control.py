import numpy as np
import pandas as pd
from config import QUALITY_CONTROL


class QualityController:
    def __init__(self):
        self.qc_rules = QUALITY_CONTROL
        self.quality_flags = {}

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
        df['temperature_assimilated'] = df['temperature'].copy()
        df['salinity_assimilated'] = df['salinity'].copy()
        df['current_speed_assimilated'] = df['current_speed'].copy()

        for col, assimilated_col in [
            ('temperature', 'temperature_assimilated'),
            ('salinity', 'salinity_assimilated'),
            ('current_speed', 'current_speed_assimilated')
        ]:
            bad_mask = df['quality_flag'].isin(['bad', 'missing'])
            if bad_mask.any() and 'depth' in df.columns:
                df_sorted = df.sort_values('depth').reset_index(drop=True)
                values = df_sorted[col].copy()
                mask = df_sorted['quality_flag'].isin(['bad', 'missing'])
                values[mask] = np.nan
                interpolated = values.interpolate(method='linear', limit_direction='both')
                df.loc[df_sorted.index, assimilated_col] = interpolated.values

        df['data_assimilated'] = df['quality_flag'].isin(['bad', 'missing'])

        return df

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


