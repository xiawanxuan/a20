import sys
sys.path.insert(0, '.')

import numpy as np
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController


def test_interpolation_methods_directly():
    fetcher = BuoyDataFetcher(use_simulation=True)
    raw_data = fetcher.fetch_realtime_data('46001')
    df_sorted = raw_data.sort_values('depth').reset_index(drop=True)

    depths = df_sorted['depth'].values
    temps = df_sorted['temperature'].values

    n_total = len(depths)
    n_missing = 20
    missing_indices = np.random.choice(n_total, n_missing, replace=False)
    missing_indices = np.sort(missing_indices)

    good_indices = np.setdiff1d(np.arange(n_total), missing_indices)
    good_depths = depths[good_indices]
    good_temps = temps[good_indices]
    bad_depths = depths[missing_indices]
    true_temps = temps[missing_indices]

    print(f"总数据点: {n_total}")
    print(f"好数据点: {len(good_depths)}")
    print(f"待插值点: {len(bad_depths)}")
    print(f"深度范围: {depths.min():.0f} - {depths.max():.0f} m")
    print(f"温度范围: {temps.min():.2f} - {temps.max():.2f} °C")
    print()

    qc = QualityController()

    methods = ['linear', 'spline', 'optimal', 'gaussian']
    results = {}

    for method in methods:
        print(f"=== {method.upper()} ===")
        qc.assimilation_method = method

        from scipy import interpolate

        if method == 'linear':
            f = interpolate.interp1d(good_depths, good_temps, kind='linear', fill_value='extrapolate')
            interpolated = f(bad_depths)
        elif method == 'spline':
            try:
                from scipy.interpolate import Akima1DInterpolator
                akima = Akima1DInterpolator(good_depths, good_temps)
                interpolated = akima(bad_depths)
                print(f"  使用 Akima 插值")
            except Exception as e:
                print(f"  Akima 失败: {e}")
                interpolated = qc._linear_interpolation(good_depths, good_temps, bad_depths)
        elif method == 'optimal':
            interpolated = qc._optimal_interpolation(good_depths, good_temps, bad_depths, 'temperature')
        elif method == 'gaussian':
            interpolated = qc._gaussian_process_interpolation(good_depths, good_temps, bad_depths)

        rmse = np.sqrt(np.mean((true_temps - interpolated) ** 2))
        mae = np.mean(np.abs(true_temps - interpolated))

        print(f"  RMSE: {rmse:.6f} °C")
        print(f"  MAE:  {mae:.6f} °C")
        print(f"  范围:  {interpolated.min():.3f} - {interpolated.max():.3f} °C")
        print(f"  真实范围: {true_temps.min():.3f} - {true_temps.max():.3f} °C")
        print()

        results[method] = {
            'interpolated': interpolated,
            'rmse': rmse,
            'mae': mae,
        }

    print("=== 方法对比 ===")
    print(f"{'方法':<12} {'RMSE':<12} {'MAE':<12} {'相对改善':<12}")
    print("-" * 48)
    baseline_rmse = results['linear']['rmse']
    for method in methods:
        r = results[method]
        improvement = (baseline_rmse - r['rmse']) / baseline_rmse * 100
        print(f"{method:<12} {r['rmse']:<12.6f} {r['mae']:<12.6f} {improvement:>+10.2f}%")


if __name__ == '__main__':
    test_interpolation_methods_directly()
