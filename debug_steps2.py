import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
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
all_temp_orig = df_corrupted['temperature'].values
all_sal_orig = df_corrupted['salinity'].values

good_mask = np.ones(n_points, dtype=bool)
good_mask[anomaly_indices] = False

good_depths = all_depths[good_mask]
good_temp = all_temp_orig[good_mask]
good_sal = all_sal_orig[good_mask]

bad_depths = all_depths[~good_mask]
true_temp = raw_data.loc[~good_mask, 'temperature'].values
true_sal = raw_data.loc[~good_mask, 'salinity'].values
corrupted_temp = all_temp_orig[~good_mask]
corrupted_sal = all_sal_orig[~good_mask]

print(f"好数据点: {good_depths.size}, 待插值点: {bad_depths.size}")
print(f"污染数据温度RMSE: {np.sqrt(np.mean((true_temp - corrupted_temp)**2)):.4f}")
print()

print("=" * 60)
print("Step 1: 基础插值方法对比")
print("=" * 60)

linear_temp = np.interp(bad_depths, good_depths, good_temp)
print(f"线性插值:       {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f} °C")

try:
    akima = Akima1DInterpolator(good_depths, good_temp)
    akima_temp = akima(bad_depths)
    akima_temp = np.where(np.isfinite(akima_temp), akima_temp, linear_temp)
    print(f"Akima插值:      {np.sqrt(np.mean((true_temp - akima_temp)**2)):.4f} °C")
except Exception as e:
    print(f"Akima失败: {e}")
    akima_temp = linear_temp

try:
    pchip = PchipInterpolator(good_depths, good_temp)
    pchip_temp = pchip(bad_depths)
    pchip_temp = np.where(np.isfinite(pchip_temp), pchip_temp, linear_temp)
    print(f"Pchip插值:      {np.sqrt(np.mean((true_temp - pchip_temp)**2)):.4f} °C")
except Exception as e:
    print(f"Pchip失败: {e}")
    pchip_temp = linear_temp

# 各种加权组合
weights_list = [
    (0.5, 0.3, 0.2, "0.5A+0.3P+0.2L"),
    (0.3, 0.3, 0.4, "0.3A+0.3P+0.4L"),
    (0.4, 0.2, 0.4, "0.4A+0.2P+0.4L"),
    (0.7, 0.0, 0.3, "0.7A+0.0P+0.3L"),
    (0.0, 0.7, 0.3, "0.0A+0.7P+0.3L"),
]

print()
print("加权融合:")
for wa, wp, wl, label in weights_list:
    combined = wa * akima_temp + wp * pchip_temp + wl * linear_temp
    rmse = np.sqrt(np.mean((true_temp - combined)**2))
    print(f"  {label}: {rmse:.4f} °C")

# 高斯平滑
print()
print("Step 2: 加入高斯平滑")
for alpha in [0.1, 0.2, 0.3, 0.5]:
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        combined = 0.5 * akima_temp + 0.3 * pchip_temp + 0.2 * linear_temp
        smoothed = gaussian_filter1d(combined, sigma=sigma)
        final = alpha * smoothed + (1 - alpha) * combined
        rmse = np.sqrt(np.mean((true_temp - final)**2))
        # 只打印较好的
        if rmse < 0.20:
            print(f"  alpha={alpha:.1f}, sigma={sigma:.1f}: {rmse:.4f} °C")

print()
print("Step 3: T-S耦合影响")
# 盐度插值
linear_sal = np.interp(bad_depths, good_depths, good_sal)
akima_sal = Akima1DInterpolator(good_depths, good_sal)(bad_depths)
pchip_sal = PchipInterpolator(good_depths, good_sal)(bad_depths)
akima_sal = np.where(np.isfinite(akima_sal), akima_sal, linear_sal)
pchip_sal = np.where(np.isfinite(pchip_sal), pchip_sal, linear_sal)

combined_temp = 0.5 * akima_temp + 0.3 * pchip_temp + 0.2 * linear_temp
combined_sal = 0.5 * akima_sal + 0.3 * pchip_sal + 0.2 * linear_sal

smoothed_temp = gaussian_filter1d(combined_temp, sigma=1.0)
temp_smooth = 0.2 * smoothed_temp + 0.8 * combined_temp

print(f"  基础温度RMSE: {np.sqrt(np.mean((true_temp - temp_smooth)**2)):.4f}")
print(f"  基础盐度RMSE: {np.sqrt(np.mean((true_sal - combined_sal)**2)):.4f}")

# T-S耦合
def ts_linear(t, a, b):
    return a + b * t

try:
    popt, _ = curve_fit(ts_linear, good_temp, good_sal, maxfev=10000)
    print(f"  T-S关系: S = {popt[0]:.4f} + {popt[1]:.4f} * T")

    for alpha_sal in [0.2, 0.4, 0.6, 0.8]:
        for alpha_temp in [0.1, 0.2, 0.3, 0.5]:
            predicted_sal = ts_linear(temp_smooth, *popt)
            sal_corrected = alpha_sal * predicted_sal + (1 - alpha_sal) * combined_sal

            predicted_temp = (sal_corrected - popt[0]) / popt[1] if popt[1] != 0 else temp_smooth
            temp_corrected = alpha_temp * predicted_temp + (1 - alpha_temp) * temp_smooth

            temp_rmse = np.sqrt(np.mean((true_temp - temp_corrected)**2))
            sal_rmse = np.sqrt(np.mean((true_sal - sal_corrected)**2))

            if temp_rmse < 0.20 and sal_rmse < 0.10:
                print(f"    a_sal={alpha_sal:.1f}, a_temp={alpha_temp:.1f}: T={temp_rmse:.4f}, S={sal_rmse:.4f}")

except Exception as e:
    print(f"  T-S耦合失败: {e}")

print()
print("Step 4: 背景场约束影响")

# 计算背景场
window = min(11, len(good_temp) // 2 * 2 - 1)
if window < 5:
    window = 5
if window % 2 == 0:
    window += 1
polyorder = min(2, window - 1)

try:
    bg_good = savgol_filter(good_temp, window, polyorder)
except Exception:
    bg_good = gaussian_filter1d(good_temp, sigma=2.0)

bg_all = np.interp(all_depths, good_depths, bg_good)
bg_bad = np.interp(bad_depths, all_depths, bg_all)

print(f"  背景场RMSE (vs真实): {np.sqrt(np.mean((true_temp - bg_bad)**2)):.4f}")

base_temp = temp_smooth
for max_dev in [0.5, 1.0, 1.5, 2.0, 3.0]:
    temp_constrained = base_temp.copy()
    deviation = temp_constrained - bg_bad
    too_far = np.abs(deviation) > max_dev
    temp_constrained[too_far] = bg_bad[too_far] + np.sign(deviation[too_far]) * max_dev
    rmse = np.sqrt(np.mean((true_temp - temp_constrained)**2))
    print(f"  max_dev={max_dev:.1f}: RMSE={rmse:.4f} (修正了{too_far.sum()}个点)")

print()
print("=" * 60)
print("结论:")
print(f"  线性插值 RMSE:   {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")
print(f"  Akima插值 RMSE:  {np.sqrt(np.mean((true_temp - akima_temp)**2)):.4f}")
print(f"  Pchip插值 RMSE:  {np.sqrt(np.mean((true_temp - pchip_temp)**2)):.4f}")
