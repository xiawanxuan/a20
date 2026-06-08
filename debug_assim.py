import sys
sys.path.insert(0, '.')

import numpy as np
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController


def debug_assimilation():
    fetcher = BuoyDataFetcher(use_simulation=True)
    raw_data = fetcher.fetch_realtime_data('46001')

    print("原始数据统计:")
    print(f"  temperature: {raw_data['temperature'].min():.3f} ~ {raw_data['temperature'].max():.3f}")
    print(f"  salinity: {raw_data['salinity'].min():.3f} ~ {raw_data['salinity'].max():.3f}")
    print(f"  current_speed: {raw_data['current_speed'].min():.4f} ~ {raw_data['current_speed'].max():.4f}")

    df_corrupted = raw_data.copy()
    n_anomalies = 20
    anomaly_indices = np.random.choice(len(df_corrupted), n_anomalies, replace=False)

    for idx in anomaly_indices:
        df_corrupted.loc[idx, 'temperature'] += np.random.uniform(-3, 3)
        df_corrupted.loc[idx, 'salinity'] += np.random.uniform(-2, 2)
        df_corrupted.loc[idx, 'current_speed'] += np.random.uniform(0, 1.5)

    print("\n异常数据统计 (加入异常后):")
    print(f"  temperature: {df_corrupted['temperature'].min():.3f} ~ {df_corrupted['temperature'].max():.3f}")
    print(f"  salinity: {df_corrupted['salinity'].min():.3f} ~ {df_corrupted['salinity'].max():.3f}")
    print(f"  current_speed: {df_corrupted['current_speed'].min():.4f} ~ {df_corrupted['current_speed'].max():.4f}")

    methods = ['linear', 'spline', 'gaussian']
    for method in methods:
        print(f"\n--- {method.upper()} 方法 ---")
        qc = QualityController(assimilation_method=method)
        qc_data = qc.run_full_qc(df_corrupted)

        print(f"  quality_flag 分布:")
        print(f"    {qc_data['quality_flag'].value_counts().to_dict()}")

        assim_mask = qc_data['data_assimilated']
        print(f"  assimilated 点数: {assim_mask.sum()}")

        if assim_mask.sum() > 0:
            print(f"  同化后温度范围: {qc_data.loc[assim_mask, 'temperature_assimilated'].min():.3f} ~ {qc_data.loc[assim_mask, 'temperature_assimilated'].max():.3f}")
            print(f"  同化后盐度范围: {qc_data.loc[assim_mask, 'salinity_assimilated'].min():.3f} ~ {qc_data.loc[assim_mask, 'salinity_assimilated'].max():.3f}")
            print(f"  同化后流速范围: {qc_data.loc[assim_mask, 'current_speed_assimilated'].min():.4f} ~ {qc_data.loc[assim_mask, 'current_speed_assimilated'].max():.4f}")

            true_temp = raw_data.loc[assim_mask, 'temperature'].values
            assim_temp = qc_data.loc[assim_mask, 'temperature_assimilated'].values
            temp_rmse = np.sqrt(np.mean((true_temp - assim_temp) ** 2))
            print(f"  温度RMSE: {temp_rmse:.4f}")

            true_speed = raw_data.loc[assim_mask, 'current_speed'].values
            assim_speed = qc_data.loc[assim_mask, 'current_speed_assimilated'].values
            speed_rmse = np.sqrt(np.mean((true_speed - assim_speed) ** 2))
            print(f"  流速RMSE: {speed_rmse:.4f}")

            print(f"  真实流速范围: {true_speed.min():.4f} ~ {true_speed.max():.4f}")
            print(f"  同化流速范围: {assim_speed.min():.4f} ~ {assim_speed.max():.4f}")

            if method == 'linear':
                print("  [DEBUG] 检查同化值是否等于原始值:")
                orig_speed = df_corrupted.loc[assim_mask, 'current_speed'].values
                print(f"    原始异常值范围: {orig_speed.min():.4f} ~ {orig_speed.max():.4f}")
                print(f"    是否与同化值相同: {np.allclose(orig_speed, assim_speed)}")


if __name__ == '__main__':
    debug_assimilation()
