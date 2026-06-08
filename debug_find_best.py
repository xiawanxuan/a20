import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from data_acquisition import BuoyDataFetcher

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

all_depths = df_corrupted['depth'].values
good_mask = np.ones(n_points, dtype=bool)
good_mask[anomaly_indices] = False

good_depths = all_depths[good_mask]
good_temp = df_corrupted.loc[good_mask, 'temperature'].values
good_sal = df_corrupted.loc[good_mask, 'salinity'].values
bad_depths = all_depths[~good_mask]
true_temp = raw_data.loc[~good_mask, 'temperature'].values
true_sal = raw_data.loc[~good_mask, 'salinity'].values

print(f"好数据: {good_depths.size}, 待插值: {bad_depths.size}")
print(f"污染数据RMSE: {np.sqrt(np.mean((true_temp - df_corrupted.loc[~good_mask,'temperature'].values)**2)):.4f}")
print()

# 各种基础插值
linear_temp = np.interp(bad_depths, good_depths, good_temp)
print(f"1. 线性插值:           {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")

try:
    akima = Akima1DInterpolator(good_depths, good_temp)
    akima_temp = akima(bad_depths)
    akima_temp = np.where(np.isfinite(akima_temp), akima_temp, linear_temp)
    print(f"2. Akima样条:          {np.sqrt(np.mean((true_temp - akima_temp)**2)):.4f}")
except Exception as e:
    print(f"   Akima失败: {e}")
    akima_temp = linear_temp

try:
    pchip = PchipInterpolator(good_depths, good_temp)
    pchip_temp = pchip(bad_depths)
    pchip_temp = np.where(np.isfinite(pchip_temp), pchip_temp, linear_temp)
    print(f"3. Pchip保形:          {np.sqrt(np.mean((true_temp - pchip_temp)**2)):.4f}")
except Exception as e:
    print(f"   Pchip失败: {e}")
    pchip_temp = linear_temp

# 背景场
window = min(11, len(good_temp) // 2 * 2 - 1)
if window < 5:
    window = 5
if window % 2 == 0:
    window += 1
bg_good = savgol_filter(good_temp, window, min(2, window - 1))
bg_bad = np.interp(bad_depths, good_depths, bg_good)
print(f"4. 背景场(SG滤波):     {np.sqrt(np.mean((true_temp - bg_bad)**2)):.4f}")

# 加权组合
print()
print("各种加权组合 (温度RMSE):")
best_rmse = 999
best_combo = ""

weights = [
    (1.0, 0.0, 0.0, "1.0线性"),
    (0.0, 1.0, 0.0, "1.0Akima"),
    (0.0, 0.0, 1.0, "1.0Pchip"),
    (0.3, 0.7, 0.0, "0.3线+0.7Aki"),
    (0.5, 0.5, 0.0, "0.5线+0.5Aki"),
    (0.7, 0.3, 0.0, "0.7线+0.3Aki"),
    (0.2, 0.5, 0.3, "0.2线+0.5A+0.3P"),
    (0.3, 0.4, 0.3, "0.3线+0.4A+0.3P"),
    (0.1, 0.6, 0.3, "0.1线+0.6A+0.3P"),
    (0.4, 0.4, 0.2, "0.4线+0.4A+0.2P"),
]

for wl, wa, wp, label in weights:
    combined = wl * linear_temp + wa * akima_temp + wp * pchip_temp
    rmse = np.sqrt(np.mean((true_temp - combined)**2))
    marker = " <--" if rmse < best_rmse else ""
    print(f"  {label:<15}: {rmse:.4f}{marker}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_combo = label

# 背景场融合
print()
print("背景场融合:")
best_base = 0.2 * linear_temp + 0.5 * akima_temp + 0.3 * pchip_temp
for w_bg in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    combined = (1 - w_bg) * best_base + w_bg * bg_bad
    rmse = np.sqrt(np.mean((true_temp - combined)**2))
    marker = " <--" if rmse < best_rmse else ""
    print(f"  w_bg={w_bg:.2f}: {rmse:.4f}{marker}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_combo = f"bg={w_bg:.2f}"

# 高斯平滑
print()
print("加入高斯平滑 (基于最佳组合):")
best_interp = best_base.copy()
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        smoothed = gaussian_filter1d(best_interp, sigma=sigma)
        final = alpha * smoothed + (1 - alpha) * best_interp
        rmse = np.sqrt(np.mean((true_temp - final)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_combo = f"平滑a={alpha:.1f},s={sigma:.1f}"

print()
print("=" * 50)
print(f"最佳组合: {best_combo}")
print(f"最佳RMSE: {best_rmse:.4f} °C")
print(f"线性基准: {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f} °C")
improvement = (np.sqrt(np.mean((true_temp - linear_temp)**2)) - best_rmse) / np.sqrt(np.mean((true_temp - linear_temp)**2)) * 100
print(f"提升幅度: {improvement:+.1f}%")
