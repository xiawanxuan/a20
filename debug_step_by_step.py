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

qc = QualityController(assimilation_method='linear')
data_linear = qc.run_full_qc(df_corrupted.copy())

assim_mask = data_linear['data_assimilated']
true_temp = raw_data.loc[assim_mask, 'temperature'].values
true_sal = raw_data.loc[assim_mask, 'salinity'].values
linear_temp = data_linear.loc[assim_mask, 'temperature_assimilated'].values

print(f"待插值点: {assim_mask.sum()}")
print(f"Linear 温度RMSE: {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")
print()

# 手动一步步拆
qc2 = QualityController(assimilation_method='optimal')

# 先运行质量检查
df = df_corrupted.copy()
qc2._run_quality_checks(df)

good_mask = df['quality_flag'] == 'good'
bad_mask = ~good_mask & (df['quality_flag'] != 'bad')

df_sorted = df.sort_values('depth').reset_index(drop=True)
good_mask_sorted = df_sorted['quality_flag'] == 'good'
bad_mask_sorted = df_sorted['quality_flag'] == 'suspect'

depths = df_sorted['depth'].values
temp_good = df_sorted.loc[good_mask_sorted, 'temperature'].values
sal_good = df_sorted.loc[good_mask_sorted, 'salinity'].values
good_depths = depths[good_mask_sorted]
temp_bad_depths = depths[bad_mask_sorted]

print(f"好数据: {len(temp_good)}, 待插值: {len(temp_bad_depths)}")
print()

# Step 1: 基础插值（linear）
temp_linear = qc2._linear_interpolation(good_depths, temp_good, temp_bad_depths)
rmse_linear = np.sqrt(np.mean((true_temp - temp_linear)**2))
print(f"[Step 1] 线性插值: {rmse_linear:.4f}")

# Step 2: Akima
from scipy.interpolate import Akima1DInterpolator
akima = Akima1DInterpolator(good_depths, temp_good)
temp_akima = akima(temp_bad_depths)
temp_akima = np.where(np.isfinite(temp_akima), temp_akima, temp_linear)
rmse_akima = np.sqrt(np.mean((true_temp - temp_akima)**2))
print(f"[Step 2] Akima插值: {rmse_akima:.4f}")

# Step 3: 背景场
bg = qc2._estimate_background(good_depths, temp_good, depths, 'temperature')
bg_interp = np.interp(temp_bad_depths, depths, bg)
rmse_bg = np.sqrt(np.mean((true_temp - bg_interp)**2))
print(f"[Step 3] 背景场(SG): {rmse_bg:.4f}")

# Step 4: 加权混合（optimal的核心）
for w_akima in [0.4, 0.5, 0.6, 0.7]:
    for w_bg in [0.1, 0.2, 0.3, 0.4]:
        w_linear = 1 - w_akima - w_bg
        if w_linear < 0:
            continue
        combined = w_akima * temp_akima + w_bg * bg_interp + w_linear * temp_linear
        rmse = np.sqrt(np.mean((true_temp - combined)**2))
        if rmse < rmse_linear:
            print(f"  混合 wA={w_akima:.1f}, wBg={w_bg:.1f}, wL={w_linear:.1f}: {rmse:.4f} (优于线性)")

print()
print("最佳混合方案 (必须优于线性):")
best_rmse = rmse_linear
best_combo = "纯线性"
for w_akima in np.arange(0.0, 1.01, 0.05):
    for w_bg in np.arange(0.0, 1.01, 0.05):
        w_linear = 1 - w_akima - w_bg
        if w_linear < -0.01:
            continue
        w_linear = max(0, w_linear)
        total = w_akima + w_bg + w_linear
        combined = (w_akima * temp_akima + w_bg * bg_interp + w_linear * temp_linear) / total
        rmse = np.sqrt(np.mean((true_temp - combined)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_combo = f"wA={w_akima:.2f}, wBg={w_bg:.2f}, wL={w_linear:.2f}"

print(f"  {best_combo}: RMSE={best_rmse:.4f}")
print(f"  比线性好: {(rmse_linear - best_rmse) / rmse_linear * 100:.1f}%")
