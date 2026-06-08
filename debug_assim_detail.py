import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
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

print(f"异常点索引: {anomaly_indices[:10]} ...")
print(f"总异常点: {len(anomaly_indices)}")
print()

# 测试 linear 方法
qc_linear = QualityController(assimilation_method='linear')
data_linear = qc_linear.run_full_qc(df_corrupted)

assim_mask = data_linear['data_assimilated']
print(f"需要同化的点数: {assim_mask.sum()}")
print(f"质量标记: {data_linear['quality_flag'].value_counts().to_dict()}")
print()

true_temp = raw_data.loc[assim_mask, 'temperature'].values
true_sal = raw_data.loc[assim_mask, 'salinity'].values

linear_temp = data_linear.loc[assim_mask, 'temperature_assimilated'].values
linear_sal = data_linear.loc[assim_mask, 'salinity_assimilated'].values

print(f"Linear 方法:")
print(f"  温度 RMSE: {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")
print(f"  盐度 RMSE: {np.sqrt(np.mean((true_sal - linear_sal)**2)):.4f}")
print()

# 测试 spline 方法
qc_spline = QualityController(assimilation_method='spline')
data_spline = qc_spline.run_full_qc(df_corrupted)

spline_temp = data_spline.loc[assim_mask, 'temperature_assimilated'].values
spline_sal = data_spline.loc[assim_mask, 'salinity_assimilated'].values

print(f"Spline 方法:")
print(f"  温度 RMSE: {np.sqrt(np.mean((true_temp - spline_temp)**2)):.4f}")
print(f"  盐度 RMSE: {np.sqrt(np.mean((true_sal - spline_sal)**2)):.4f}")
print()

# 测试 optimal 方法
qc_optimal = QualityController(assimilation_method='optimal')
data_optimal = qc_optimal.run_full_qc(df_corrupted)

optimal_temp = data_optimal.loc[assim_mask, 'temperature_assimilated'].values
optimal_sal = data_optimal.loc[assim_mask, 'salinity_assimilated'].values

print(f"Optimal 方法:")
print(f"  温度 RMSE: {np.sqrt(np.mean((true_temp - optimal_temp)**2)):.4f}")
print(f"  盐度 RMSE: {np.sqrt(np.mean((true_sal - optimal_sal)**2)):.4f}")
print(f"  温度范围: {optimal_temp.min():.3f} - {optimal_temp.max():.3f}")
print(f"  真实范围: {true_temp.min():.3f} - {true_temp.max():.3f}")
print(f"  Linear范围: {linear_temp.min():.3f} - {linear_temp.max():.3f}")
print()

# 输出前10个同化点的详细对比
print("前10个同化点详细对比:")
print(f"{'深度':>8} {'真实温度':>10} {'Linear':>10} {'Spline':>10} {'Optimal':>10} {'Linear误':>10} {'Spline误':>10} {'Optimal误':>10}")

assim_depths = data_linear.loc[assim_mask, 'depth'].values
for i in range(min(10, len(assim_depths))):
    print(f"{assim_depths[i]:>8.1f} {true_temp[i]:>10.3f} {linear_temp[i]:>10.3f} {spline_temp[i]:>10.3f} {optimal_temp[i]:>10.3f} {abs(linear_temp[i]-true_temp[i]):>10.3f} {abs(spline_temp[i]-true_temp[i]):>10.3f} {abs(optimal_temp[i]-true_temp[i]):>10.3f}")

# 看看大误差的点
print()
print("Optimal 方法误差最大的5个点:")
errors = np.abs(optimal_temp - true_temp)
sorted_idx = np.argsort(errors)[::-1][:5]
for i in sorted_idx:
    print(f"  深度 {assim_depths[i]:6.1f}m: 真实={true_temp[i]:.3f}, 最优={optimal_temp[i]:.3f}, 误差={errors[i]:.3f}")

# 看看和linear的偏差
print()
print("Optimal vs Linear 偏差最大的5个点:")
diff = np.abs(optimal_temp - linear_temp)
sorted_idx2 = np.argsort(diff)[::-1][:5]
for i in sorted_idx2:
    print(f"  深度 {assim_depths[i]:6.1f}m: Linear={linear_temp[i]:.3f}, Optimal={optimal_temp[i]:.3f}, 偏差={diff[i]:.3f}")
