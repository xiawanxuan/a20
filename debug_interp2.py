import sys
sys.path.insert(0, '.')
import numpy as np
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController

fetcher = BuoyDataFetcher(use_simulation=True)
raw_data = fetcher.fetch_realtime_data('46001')

df_corrupted = raw_data.copy()
n_points = len(df_corrupted)
n_anomalies = int(n_points * 0.25)
np.random.seed(42)
anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
anomaly_indices = np.sort(anomaly_indices)

for idx in anomaly_indices:
    depth = df_corrupted.loc[idx, 'depth']
    temp_bias = np.random.uniform(-2.5, 2.5)
    sal_bias = np.random.uniform(-1.5, 1.5)
    if depth < 100:
        temp_bias *= 1.3
    elif depth > 500:
        temp_bias *= 0.4
    df_corrupted.loc[idx, 'temperature'] += temp_bias
    df_corrupted.loc[idx, 'salinity'] += sal_bias

print("原始数据 vs 污染数据 (前20行):")
print(f"{'深度':>8} {'原始温度':>10} {'污染温度':>10} {'原始盐度':>10} {'污染盐度':>10}")
for i in range(min(20, n_points)):
    print(f"{raw_data.loc[i,'depth']:>8.1f} {raw_data.loc[i,'temperature']:>10.3f} {df_corrupted.loc[i,'temperature']:>10.3f} {raw_data.loc[i,'salinity']:>10.3f} {df_corrupted.loc[i,'salinity']:>10.3f}")

print(f"\n总点数: {n_points}, 异常点数: {n_anomalies}")

from scipy.interpolate import PchipInterpolator, Akima1DInterpolator

qc_linear = QualityController(assimilation_method='linear')
data_linear = qc_linear.run_full_qc(df_corrupted)

qc_spline = QualityController(assimilation_method='spline')
data_spline = qc_spline.run_full_qc(df_corrupted)

qc_optimal = QualityController(assimilation_method='optimal')
data_optimal = qc_optimal.run_full_qc(df_corrupted)

assim_mask = data_linear['data_assimilated']
print(f"\n需要同化的点数: {assim_mask.sum()}")
print(f"质量标记: {data_linear['quality_flag'].value_counts().to_dict()}")

true_temp = raw_data.loc[assim_mask, 'temperature'].values
true_sal = raw_data.loc[assim_mask, 'salinity'].values

linear_temp = data_linear.loc[assim_mask, 'temperature_assimilated'].values
spline_temp = data_spline.loc[assim_mask, 'temperature_assimilated'].values
optimal_temp = data_optimal.loc[assim_mask, 'temperature_assimilated'].values

linear_sal = data_linear.loc[assim_mask, 'salinity_assimilated'].values
spline_sal = data_spline.loc[assim_mask, 'salinity_assimilated'].values
optimal_sal = data_optimal.loc[assim_mask, 'salinity_assimilated'].values

orig_temp = df_corrupted.loc[assim_mask, 'temperature'].values
orig_sal = df_corrupted.loc[assim_mask, 'salinity'].values

print(f"\n温度 RMSE:")
print(f"  原始:   {np.sqrt(np.mean((true_temp - orig_temp)**2)):.4f}")
print(f"  线性:   {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")
print(f"  Spline: {np.sqrt(np.mean((true_temp - spline_temp)**2)):.4f}")
print(f"  Optimal:{np.sqrt(np.mean((true_temp - optimal_temp)**2)):.4f}")

print(f"\n盐度 RMSE:")
print(f"  原始:   {np.sqrt(np.mean((true_sal - orig_sal)**2)):.4f}")
print(f"  线性:   {np.sqrt(np.mean((true_sal - linear_sal)**2)):.4f}")
print(f"  Spline: {np.sqrt(np.mean((true_sal - spline_sal)**2)):.4f}")
print(f"  Optimal:{np.sqrt(np.mean((true_sal - optimal_sal)**2)):.4f}")

# 手动调试插值
good_mask = df_corrupted['quality_flag'] == 'good' if 'quality_flag' in df_corrupted.columns else None
qc_temp = QualityController(assimilation_method='linear')
qc_temp._apply_range_check(df_corrupted, 'temperature')
qc_temp._apply_statistical_check(df_corrupted, 'temperature')
qc_temp._apply_gradient_check(df_corrupted, 'temperature')

good_mask = data_linear['quality_flag'] == 'good'
good_depths = data_linear.loc[good_mask, 'depth'].values
good_temp = data_linear.loc[good_mask, 'temperature'].values
interp_depths = data_linear.loc[assim_mask, 'depth'].values

print(f"\n调试:")
print(f"  好数据点: {len(good_depths)} 个")
print(f"  待插值点: {len(interp_depths)} 个")
print(f"  好数据深度范围: {good_depths.min():.1f} - {good_depths.max():.1f}")
print(f"  待插值深度范围: {interp_depths.min():.1f} - {interp_depths.max():.1f}")

# 手动计算几种插值
linear_vals = np.interp(interp_depths, good_depths, good_temp)
print(f"\n手动线性插值 RMSE: {np.sqrt(np.mean((true_temp - linear_vals)**2)):.4f}")

try:
    pchip = PchipInterpolator(good_depths, good_temp)
    pchip_vals = pchip(interp_depths)
    pchip_vals = np.where(np.isfinite(pchip_vals), pchip_vals, linear_vals)
    print(f"Pchip插值 RMSE: {np.sqrt(np.mean((true_temp - pchip_vals)**2)):.4f}")
    print(f"  Pchip 范围: {pchip_vals.min():.3f} - {pchip_vals.max():.3f}")
    print(f"  真实范围: {true_temp.min():.3f} - {true_temp.max():.3f}")
except Exception as e:
    print(f"Pchip失败: {e}")

try:
    akima = Akima1DInterpolator(good_depths, good_temp)
    akima_vals = akima(interp_depths)
    akima_vals = np.where(np.isfinite(akima_vals), akima_vals, linear_vals)
    print(f"Akima插值 RMSE: {np.sqrt(np.mean((true_temp - akima_vals)**2)):.4f}")
    print(f"  Akima 范围: {akima_vals.min():.3f} - {akima_vals.max():.3f}")
except Exception as e:
    print(f"Akima失败: {e}")

# 看看数据同化后的全部值
print(f"\n同化后温度对比 (深度从0到500m):")
shallow_mask = interp_depths <= 500
print(f"  浅层点 {shallow_mask.sum()} 个:")
print(f"    线性 RMSE:  {np.sqrt(np.mean((true_temp[shallow_mask] - linear_temp[shallow_mask])**2)):.4f}")
print(f"    Spline RMSE:{np.sqrt(np.mean((true_temp[shallow_mask] - spline_temp[shallow_mask])**2)):.4f}")
print(f"    Optimal RMSE: {np.sqrt(np.mean((true_temp[shallow_mask] - optimal_temp[shallow_mask])**2)):.4f}")

deep_mask = interp_depths > 500
print(f"  深层点 {deep_mask.sum()} 个:")
print(f"    线性 RMSE:  {np.sqrt(np.mean((true_temp[deep_mask] - linear_temp[deep_mask])**2)):.4f}")
print(f"    Spline RMSE:{np.sqrt(np.mean((true_temp[deep_mask] - spline_temp[deep_mask])**2)):.4f}")
print(f"    Optimal RMSE: {np.sqrt(np.mean((true_temp[deep_mask] - optimal_temp[deep_mask])**2)):.4f}")
