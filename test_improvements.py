import sys
import os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController
from visualization import OceanVisualizer
from main import DataExporter


def generate_realistic_anomalies(df, anomaly_ratio=0.15):
    df_corrupted = df.copy()
    n_points = len(df_corrupted)
    n_anomalies = int(n_points * anomaly_ratio)

    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    anomaly_indices = np.sort(anomaly_indices)

    for idx in anomaly_indices:
        depth = df_corrupted.loc[idx, 'depth']

        temp_bias = np.random.uniform(-3, 3)
        sal_bias = np.random.uniform(-2, 2)
        speed_bias = np.random.uniform(-0.8, 1.5)

        if depth < 100:
            temp_bias *= 1.5
        elif depth > 500:
            temp_bias *= 0.3
            sal_bias *= 0.5

        df_corrupted.loc[idx, 'temperature'] += temp_bias
        df_corrupted.loc[idx, 'salinity'] += sal_bias
        df_corrupted.loc[idx, 'current_speed'] += max(0, speed_bias)

    return df_corrupted, anomaly_indices


def test_assimilation_accuracy():
    print("=" * 70)
    print("数据同化精度测试 - 模拟真实海洋数据异常")
    print("=" * 70)

    fetcher = BuoyDataFetcher(use_simulation=True)
    raw_data = fetcher.fetch_realtime_data('46001')

    df_corrupted, anomaly_indices = generate_realistic_anomalies(raw_data, anomaly_ratio=0.2)

    print(f"\n总数据点: {len(raw_data)}")
    print(f"异常数据点: {len(anomaly_indices)} ({len(anomaly_indices)/len(raw_data)*100:.1f}%)")
    print(f"深度范围: {raw_data['depth'].min():.0f} - {raw_data['depth'].max():.0f} m")

    methods = ['linear', 'spline', 'optimal', 'gaussian']
    results = {}

    for method in methods:
        print(f"\n{'─' * 50}")
        print(f"方法: {method.upper()}")
        print(f"{'─' * 50}")

        qc = QualityController(assimilation_method=method)
        qc_data = qc.run_full_qc(df_corrupted)

        assim_mask = qc_data['data_assimilated']
        if assim_mask.sum() == 0:
            print("  没有数据需要同化")
            continue

        true_vals_temp = raw_data.loc[assim_mask, 'temperature'].values
        assim_vals_temp = qc_data.loc[assim_mask, 'temperature_assimilated'].values

        true_vals_sal = raw_data.loc[assim_mask, 'salinity'].values
        assim_vals_sal = qc_data.loc[assim_mask, 'salinity_assimilated'].values

        true_vals_speed = raw_data.loc[assim_mask, 'current_speed'].values
        assim_vals_speed = qc_data.loc[assim_mask, 'current_speed_assimilated'].values

        temp_rmse = np.sqrt(np.mean((true_vals_temp - assim_vals_temp) ** 2))
        temp_mae = np.mean(np.abs(true_vals_temp - assim_vals_temp))
        temp_bias = np.mean(assim_vals_temp - true_vals_temp)

        sal_rmse = np.sqrt(np.mean((true_vals_sal - assim_vals_sal) ** 2))
        sal_mae = np.mean(np.abs(true_vals_sal - assim_vals_sal))
        sal_bias = np.mean(assim_vals_sal - true_vals_sal)

        speed_rmse = np.sqrt(np.mean((true_vals_speed - assim_vals_speed) ** 2))

        original_temp = df_corrupted.loc[assim_mask, 'temperature'].values
        original_rmse = np.sqrt(np.mean((true_vals_temp - original_temp) ** 2))

        improvement = (original_rmse - temp_rmse) / original_rmse * 100 if original_rmse > 0 else 0

        print(f"  同化数据点: {assim_mask.sum()}")
        print(f"  温度:")
        print(f"    RMSE:  {temp_rmse:.4f} °C")
        print(f"    MAE:   {temp_mae:.4f} °C")
        print(f"    Bias:  {temp_bias:.4f} °C")
        print(f"  盐度:")
        print(f"    RMSE:  {sal_rmse:.4f} PSU")
        print(f"    MAE:   {sal_mae:.4f} PSU")
        print(f"    Bias:  {sal_bias:.4f} PSU")
        print(f"  海流速度:")
        print(f"    RMSE:  {speed_rmse:.4f} m/s")
        print(f"  温度改善: {improvement:.1f}% (相比原始异常数据)")

        if 'assimilation_confidence' in qc_data.columns:
            mean_conf = qc_data.loc[assim_mask, 'assimilation_confidence'].mean()
            print(f"  平均置信度: {mean_conf:.4f}")

        results[method] = {
            'temp_rmse': temp_rmse,
            'temp_mae': temp_mae,
            'sal_rmse': sal_rmse,
            'sal_mae': sal_mae,
            'speed_rmse': speed_rmse,
            'improvement_pct': improvement,
        }

    print(f"\n{'=' * 70}")
    print("方法对比总结")
    print(f"{'=' * 70}")
    print(f"{'方法':<12} {'温度RMSE':<12} {'盐度RMSE':<12} {'改善率':<12}")
    print(f"{'-' * 48}")
    for method in methods:
        if method in results:
            r = results[method]
            print(f"{method:<12} {r['temp_rmse']:<12.4f} {r['sal_rmse']:<12.4f} {r['improvement_pct']:<12.1f}%")

    best_method = min(results.keys(), key=lambda m: results[m]['temp_rmse'])
    print(f"\n最优方法: {best_method.upper()} (温度RMSE最低)")

    print("\n[OK] 数据同化精度测试完成")
    print()
    return results


def test_3d_visualization_comprehensive():
    print("=" * 70)
    print("三维可视化功能测试")
    print("=" * 70)

    fetcher = BuoyDataFetcher(use_simulation=True)
    data = fetcher.fetch_realtime_data('46001')

    qc = QualityController(assimilation_method='optimal')
    qc_data = qc.run_full_qc(data)

    viz = OceanVisualizer()

    viz_3d = [
        ('3D T-S散点图 (温盐深三维)', 'create_3d_ts_scatter'),
        ('3D 海流速度矢量图 (锥形)', 'create_3d_current_vectors'),
        ('3D 温度体积渲染 (曲面)', 'create_3d_temperature_volume'),
        ('3D 盐度体积渲染 (曲面)', 'create_3d_salinity_volume'),
        ('3D 全球浮标分布图', 'create_3d_multi_buoy_map'),
        ('3D 综合仪表板 (2x2子图)', 'create_3d_dashboard'),
    ]

    figures = {}
    for name, func_name in viz_3d:
        print(f"\n生成: {name}...")
        func = getattr(viz, func_name)
        if 'multi_buoy' in func_name:
            buoys = fetcher.fetch_buoy_list()
            surface_data = fetcher.fetch_surface_data()
            fig = func(buoys, surface_data)
        else:
            fig = func(qc_data)
        figures[name] = fig
        print(f"  [OK] 成功生成 - {type(fig).__name__}")

    print(f"\n{'=' * 70}")
    print(f"共生成 {len(figures)} 种三维可视化图表:")
    print(f"{'=' * 70}")
    for i, name in enumerate(figures.keys(), 1):
        print(f"  {i}. {name}")

    print("\n三维可视化特点:")
    print("  • 3D 可交互旋转、缩放、平移")
    print("  • 颜色映射展示数值大小")
    print("  • 悬浮显示详细数据")
    print("  • 支持深度方向的立体分布展示")
    print("  • 矢量场用锥形图直观展示流向流速")
    print("  • 体积渲染展示温盐的立体分布")

    print("\n[OK] 三维可视化测试完成")
    print()
    return figures


def export_demo_report():
    print("=" * 70)
    print("生成演示报告 (包含高级同化+3D可视化)")
    print("=" * 70)

    fetcher = BuoyDataFetcher(use_simulation=True)
    data = fetcher.fetch_realtime_data('46001')

    methods = ['linear', 'optimal']
    all_figures = {}

    viz = OceanVisualizer()

    for method in methods:
        print(f"\n使用 {method.upper()} 方法进行数据同化...")
        qc = QualityController(assimilation_method=method)
        qc_data = qc.run_full_qc(data)

        print(f"  生成3D图表...")
        fig_3d_dashboard = viz.create_3d_dashboard(qc_data)
        fig_3d_temp = viz.create_3d_temperature_volume(qc_data)
        fig_3d_current = viz.create_3d_current_vectors(qc_data)
        fig_ts_3d = viz.create_3d_ts_scatter(qc_data)

        all_figures[f'{method}_3d_dashboard'] = fig_3d_dashboard
        all_figures[f'{method}_3d_temp'] = fig_3d_temp
        all_figures[f'{method}_3d_current'] = fig_3d_current
        all_figures[f'{method}_3d_ts'] = fig_ts_3d

    exporter = DataExporter()

    print(f"\n导出所有3D可视化图表...")
    exported = {}
    for name, fig in all_figures.items():
        path = exporter.export_figure(fig, f'demo_{name}.html')
        exported[name] = path
        print(f"  ✓ {os.path.basename(path)}")

    print(f"\n共导出 {len(exported)} 个HTML文件")
    print(f"输出目录: {exporter.output_dir}")

    print("\n[OK] 演示报告生成完成")
    print()
    return exported


def main():
    print("\n" + "=" * 70)
    print("  海洋浮标数据系统 - 高级功能验证")
    print("  1. 高级数据同化 (多种算法对比)")
    print("  2. 三维立体可视化 (6种3D图表)")
    print("=" * 70 + "\n")

    try:
        results_assim = test_assimilation_accuracy()
        results_viz = test_3d_visualization_comprehensive()
        exported = export_demo_report()

        print("=" * 70)
        print("[SUCCESS] 所有高级功能验证通过!")
        print("=" * 70)
        print()
        print("改进总结:")
        print()
        print("【数据同化 - 从线性插值升级为多种高级算法】")
        print("  1. 线性插值 (linear) - 基准方法")
        print("  2. 样条插值 (spline) - 平滑曲线拟合")
        print("  3. 最优插值 (optimal) - 基于背景场+协方差分析")
        print("     • Savitzky-Golay 滤波估计背景场")
        print("     • 高斯协方差函数 (特征长度150m)")
        print("     • 观测误差/背景误差模型")
        print("     • 与样条插值加权融合 (60%最优+40%样条)")
        print("  4. 高斯过程 (gaussian) - RBF径向基函数")
        print("  5. T-S物理约束 - 热力学一致性校正")
        print("     • 二次多项式拟合温盐关系")
        print("     • 70%预测 + 30%原始插值 加权融合")
        print("  6. 置信度评估 - 基于数据点距离和密度")
        print()
        print("【三维可视化 - 从2D升级为3D立体渲染】")
        print("  1. 3D T-S散点图 - 深度/温度/盐度三维关系")
        print("  2. 3D 海流矢量图 - 锥形箭头展示u/v/w分量")
        print("  3. 3D 温度体积渲染 - 旋转曲面展示温度立体分布")
        print("  4. 3D 盐度体积渲染 - 旋转曲面展示盐度立体分布")
        print("  5. 3D 全球浮标地图 - 经纬度+海温的三维分布")
        print("  6. 3D 综合仪表板 - 2x2子图全景展示")
        print()
        print("打开 output 目录中的 HTML 文件即可查看交互式3D图表")
        print()

        return 0

    except Exception as e:
        print(f"\n[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
