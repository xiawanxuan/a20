import sys
sys.path.insert(0, '.')
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
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
qc._run_quality_checks(df_corrupted)

good_mask = df_corrupted['quality_flag'] == 'good'
bad_mask = ~good_mask
good_depths = df_corrupted.loc[good_mask, 'depth'].values
good_temp = df_corrupted.loc[good_mask, 'temperature'].values
good_sal = df_corrupted.loc[good_mask, 'salinity'].values
bad_depths = df_corrupted.loc[bad_mask, 'depth'].values
all_depths = df_corrupted['depth'].values

true_temp = raw_data.loc[bad_mask, 'temperature'].values
true_sal = raw_data.loc[bad_mask, 'salinity'].values

print(f"好数据点: {good_depths.size}, 待插值点: {bad_depths.size}")
print(f"原始温度RMSE: {np.sqrt(np.mean((true_temp - df_corrupted.loc[bad_mask,'temperature'].values)**2)):.4f}")
print()

# 步骤1: 基础插值
linear_temp = np.interp(bad_depths, good_depths, good_temp)
print(f"[Step 1] 线性插值:")
print(f"  温度RMSE: {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")

try:
    akima = Akima1DInterpolator(good_depths, good_temp)
    akima_temp = akima(bad_depths)
    akima_temp = np.where(np.isfinite(akima_temp), akima_temp, linear_temp)
    print(f"  Akima RMSE: {np.sqrt(np.mean((true_temp - akima_temp)**2)):.4f}")
except Exception as e:
    print(f"  Akima失败: {e}")
    akima_temp = linear_temp

try:
    pchip = PchipInterpolator(good_depths, good_temp)
    pchip_temp = pchip(bad_depths)
    pchip_temp = np.where(np.isfinite(pchip_temp), pchip_temp, linear_temp)
    print(f"  Pchip RMSE: {np.sqrt(np.mean((true_temp - pchip_temp)**2)):.4f}")
except Exception as e:
    print(f"  Pchip失败: {e}")
    pchip_temp = linear_temp

# 加权融合
combined_temp = 0.5 * akima_temp + 0.3 * pchip_temp + 0.2 * linear_temp
print(f"  加权融合 RMSE: {np.sqrt(np.mean((true_temp - combined_temp)**2)):.4f}")

# 高斯平滑
from scipy.ndimage import gaussian_filter1d
smoothed = gaussian_filter1d(combined_temp, sigma=1.0)
combined_smooth = 0.3 * smoothed + 0.7 * combined_temp
print(f"  +高斯平滑 RMSE: {np.sqrt(np.mean((true_temp - combined_smooth)**2)):.4f}")

print()

# 步骤2: T-S耦合
print("[Step 2] T-S耦合影响:")
# 先算盐度
linear_sal = np.interp(bad_depths, good_depths, good_sal)
akima_sal = Akima1DInterpolator(good_depths, good_sal)(bad_depths)
pchip_sal = PchipInterpolator(good_depths, good_sal)(bad_depths)
akima_sal = np.where(np.isfinite(akima_sal), akima_sal, linear_sal)
pchip_sal = np.where(np.isfinite(pchip_sal), pchip_sal, linear_sal)
combined_sal = 0.5 * akima_sal + 0.3 * pchip_sal + 0.2 * linear_sal

print(f"  盐度插值 RMSE: {np.sqrt(np.mean((true_sal - combined_sal)**2)):.4f}")

# T-S耦合
from scipy.optimize import curve_fit
def ts_linear(t, a, b):
    return a + b * t

try:
    popt, _ = curve_fit(ts_linear, good_temp, good_sal, maxfev=10000)
    predicted_sal = ts_linear(combined_smooth, *popt)
    sal_corrected = 0.6 * predicted_sal + 0.4 * combined_sal

    predicted_temp = (sal_corrected - popt[0]) / popt[1] if popt[1] != 0 else combined_smooth
    temp_corrected = 0.3 * predicted_temp + 0.7 * combined_smooth

    print(f"  +T-S耦合 温度RMSE: {np.sqrt(np.mean((true_temp - temp_corrected)**2)):.4f}")
    print(f"  +T-S耦合 盐度RMSE: {np.sqrt(np.mean((true_sal - sal_corrected)**2)):.4f}")
except Exception as e:
    print(f"  T-S耦合失败: {e}")
    temp_corrected = combined_smooth
    sal_corrected = combined_sal

print()

# 步骤3: 静力稳定度
print("[Step 3] 静力稳定度约束影响:")
# 简化版静力稳定度
def approx_density(temp, sal=34.5):
    rho0 = 1025.0
    alpha = 0.0002
    beta = 0.0007
    return rho0 * (1 - alpha * (temp - 10) + beta * (sal - 35))

# 构造完整的温度剖面（好数据+插值数据）
all_temp = np.interp(all_depths, bad_depths, temp_corrected)
all_sal = np.interp(all_depths, bad_depths, sal_corrected)

# 把好数据点也填回去
for i, d in enumerate(all_depths):
    if d in good_depths:
        idx = np.where(good_depths == d)[0][0]
        all_temp[i] = good_temp[idx]
        all_sal[i] = good_sal[idx]

density = approx_density(all_temp, all_sal)
print(f"  初始密度是否递增: {np.all(np.diff(density) >= 0)}")

# 静力稳定度调整
temp_stable = temp_corrected.copy()
sal_stable = sal_corrected.copy()

# 检查并修正不稳定层
for i in range(1, len(bad_depths) - 1):
    # 找到该深度在all_depths中的位置
    pass

print(f"  静力稳定度后温度RMSE: (未实现完整版本)")

print()

# 步骤4: 海洋物理约束（背景场）
print("[Step 4] 背景场约束影响:")
from scipy.signal import savgol_filter

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

# 应用约束
max_dev = 2.0
temp_final = temp_corrected.copy()
deviation = temp_final - bg_bad
too_far = np.abs(deviation) > max_dev
temp_final[too_far] = bg_bad[too_far] + np.sign(deviation[too_far]) * max_dev

print(f"  +背景场约束后温度RMSE: {np.sqrt(np.mean((true_temp - temp_final)**2)):.4f}")

print()
print("=" * 60)
print("总结:")
print(f"  线性插值:       {np.sqrt(np.mean((true_temp - linear_temp)**2)):.4f}")
print(f"  Akima:          {np.sqrt(np.mean((true_temp - akima_temp)**2)):.4f}")
print(f"  Pchip:          {np.sqrt(np.mean((true_temp - pchip_temp)**2)):.4f}")
print(f"  加权融合:       {np.sqrt(np.mean((true_temp - combined_temp)**2)):.4f}")
print(f"  +平滑:          {np.sqrt(np.mean((true_temp - combined_smooth)**2)):.4f}")
print(f"  +T-S耦合:       {np.sqrt(np.mean((true_temp - temp_corrected)**2)):.4f}")
print(f"  +背景场约束:    {np.sqrt(np.mean((true_temp - temp_final)**2)):.4f}")
