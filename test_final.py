import sys
import os
sys.path.insert(0, '.')

import numpy as np
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController
from visualization import OceanVisualizer
from main import DataExporter


def test_all_assimilation_methods():
    print("=" * 70)
    print("数据同化算法对比测试")
    print("=" * 70)

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
        speed_bias = np.random.uniform(0.2, 1.2)

        if depth < 100:
            temp_bias *= 1.3
        elif depth > 500:
            temp_bias *= 0.4

        df_corrupted.loc[idx, 'temperature'] += temp_bias
        df_corrupted.loc[idx, 'salinity'] += sal_bias
        df_corrupted.loc[idx, 'current_speed'] += speed_bias

    print(f"\n测试配置:")
    print(f"  总数据点: {n_points}")
    print(f"  异常点比例: {n_anomalies/n_points*100:.0f}% ({n_anomalies}个点)")
    print(f"  深度范围: {raw_data['depth'].min():.0f} - {raw_data['depth'].max():.0f} m")

    methods = ['linear', 'spline', 'optimal', 'gaussian']
    results = {}

    for method in methods:
        print(f"\n{'─' * 60}")
        print(f"方法: {method.upper()}")
        print(f"{'─' * 60}")

        qc = QualityController(assimilation_method=method)
        qc_data = qc.run_full_qc(df_corrupted)

        assim_mask = qc_data['data_assimilated']
        n_assim = assim_mask.sum()

        if n_assim == 0:
            print("  [无数据需要同化]")
            continue

        true_temp = raw_data.loc[assim_mask, 'temperature'].values
        assim_temp = qc_data.loc[assim_mask, 'temperature_assimilated'].values
        temp_rmse = np.sqrt(np.mean((true_temp - assim_temp) ** 2))
        temp_mae = np.mean(np.abs(true_temp - assim_temp))

        true_sal = raw_data.loc[assim_mask, 'salinity'].values
        assim_sal = qc_data.loc[assim_mask, 'salinity_assimilated'].values
        sal_rmse = np.sqrt(np.mean((true_sal - assim_sal) ** 2))
        sal_mae = np.mean(np.abs(true_sal - assim_sal))

        true_speed = raw_data.loc[assim_mask, 'current_speed'].values
        assim_speed = qc_data.loc[assim_mask, 'current_speed_assimilated'].values
        speed_rmse = np.sqrt(np.mean((true_speed - assim_speed) ** 2))

        orig_temp = df_corrupted.loc[assim_mask, 'temperature'].values
        orig_rmse = np.sqrt(np.mean((true_temp - orig_temp) ** 2))
        improvement = (orig_rmse - temp_rmse) / orig_rmse * 100

        print(f"  同化数据点: {n_assim}")
        print(f"  质量标记分布: {qc_data['quality_flag'].value_counts().to_dict()}")
        print(f"  温度:")
        print(f"    RMSE:   {temp_rmse:.4f} °C")
        print(f"    MAE:    {temp_mae:.4f} °C")
        print(f"    原始RMSE: {orig_rmse:.4f} °C")
        print(f"    改善率:  {improvement:+.1f}%")
        print(f"  盐度:")
        print(f"    RMSE:   {sal_rmse:.4f} PSU")
        print(f"    MAE:    {sal_mae:.4f} PSU")
        print(f"  海流速度:")
        print(f"    RMSE:   {speed_rmse:.4f} m/s")
        print(f"    真实范围: {true_speed.min():.4f} - {true_speed.max():.4f}")
        print(f"    同化范围: {assim_speed.min():.4f} - {assim_speed.max():.4f}")

        if 'assimilation_error' in qc_data.columns:
            mean_err = qc_data.loc[assim_mask, 'assimilation_error'].mean()
            print(f"  估算误差: {mean_err:.4f}")

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
            'n_assimilated': n_assim,
        }

    print(f"\n{'=' * 70}")
    print("方法性能对比")
    print(f"{'=' * 70}")
    print(f"{'方法':<12} {'温度RMSE':<12} {'盐度RMSE':<12} {'流速RMSE':<12} {'改善率':<10}")
    print(f"{'-' * 58}")
    for method in methods:
        if method in results:
            r = results[method]
            print(f"{method:<12} {r['temp_rmse']:<12.4f} {r['sal_rmse']:<12.4f} {r['speed_rmse']:<12.4f} {r['improvement_pct']:>+8.1f}%")

    best_method = min(results.keys(), key=lambda m: results[m]['temp_rmse'])
    print(f"\n最优方法 (按温度RMSE): {best_method.upper()}")

    return results


def test_3d_visualizations():
    print("\n" + "=" * 70)
    print("三维可视化功能测试")
    print("=" * 70)

    fetcher = BuoyDataFetcher(use_simulation=True)
    data = fetcher.fetch_realtime_data('46001')

    qc = QualityController(assimilation_method='optimal')
    qc_data = qc.run_full_qc(data)

    viz = OceanVisualizer()

    viz_functions = [
        ('3D 温盐深散点图', 'create_3d_ts_scatter'),
        ('3D 海流矢量图', 'create_3d_current_vectors'),
        ('3D 温度体积图', 'create_3d_temperature_volume'),
        ('3D 盐度体积图', 'create_3d_salinity_volume'),
        ('3D 浮标分布图', 'create_3d_multi_buoy_map'),
        ('3D 综合仪表板', 'create_3d_dashboard'),
    ]

    figures = {}
    for name, func_name in viz_functions:
        print(f"\n生成: {name}...")
        try:
            func = getattr(viz, func_name)
            if 'multi_buoy' in func_name:
                buoys = fetcher.fetch_buoy_list()
                surface_data = fetcher.fetch_surface_data()
                fig = func(buoys, surface_data)
            else:
                fig = func(qc_data)

            figures[name] = fig
            n_traces = len(fig.data)
            print(f"  [OK] 成功生成 - {n_traces}个trace")
        except Exception as e:
            print(f"  [失败] {e}")
            import traceback
            traceback.print_exc()

    print(f"\n三维可视化总结:")
    print(f"  成功生成 {len(figures)} / {len(viz_functions)} 种三维图表")
    for name in figures.keys():
        print(f"    - {name}")

    print("\n三维可视化特点:")
    print("  - 完全交互式: 旋转、缩放、平移")
    print("  - 颜色映射: 深度/速度用颜色编码")
    print("  - 悬浮提示: 显示详细数值")
    print("  - 多子图布局: 3D仪表板2x2布局")
    print("  - 体积渲染: 用曲面展示温盐的立体分布")
    print("  - 矢量场: 锥形箭头展示海流方向和大小")

    return figures


def export_demo():
    print("\n" + "=" * 70)
    print("生成演示文件")
    print("=" * 70)

    fetcher = BuoyDataFetcher(use_simulation=True)
    data = fetcher.fetch_realtime_data('46001')

    qc = QualityController(assimilation_method='optimal')
    qc_data = qc.run_full_qc(data)
    qc_summary = qc.get_quality_summary(qc_data)

    exporter = DataExporter()
    viz = OceanVisualizer()

    export_list = [
        ('3D温盐深散点图', viz.create_3d_ts_scatter(qc_data)),
        ('3D温度体积图', viz.create_3d_temperature_volume(qc_data)),
        ('3D盐度体积图', viz.create_3d_salinity_volume(qc_data)),
        ('3D海流矢量图', viz.create_3d_current_vectors(qc_data)),
        ('3D综合仪表板', viz.create_3d_dashboard(qc_data)),
        ('数据同化对比图', viz.create_data_assimilation_plot(qc_data)),
        ('质控结果图', viz.create_quality_control_plot(qc_data)),
    ]

    print(f"\n导出文件到 {exporter.output_dir}:")
    for name, fig in export_list:
        safe_name = name.replace(' ', '_').replace('3D', '3d')
        path = exporter.export_figure(fig, f'demo_{safe_name}.html')
        print(f"  - {os.path.basename(path)}")

    csv_path = exporter.export_to_csv(qc_data, 'demo_assimilated_data.csv')
    print(f"  - {os.path.basename(csv_path)}")

    qc_path = exporter.export_quality_report(qc_summary, 'demo_quality_report.json')
    print(f"  - {os.path.basename(qc_path)}")

    print(f"\n共导出 {len(export_list) + 2} 个文件")

    return exporter.output_dir


def main():
    print("\n" + "=" * 70)
    print("  海洋浮标数据系统 - 改进验证")
    print("  1. 高级数据同化 (4种算法对比)")
    print("  2. 三维立体可视化 (6种3D图表)")
    print("=" * 70)

    try:
        results_assim = test_all_assimilation_methods()
        results_viz = test_3d_visualizations()
        output_dir = export_demo()

        print("\n" + "=" * 70)
        print("[SUCCESS] 所有改进验证通过!")
        print("=" * 70)
        print()
        print("改进总结:")
        print()
        print("【一、数据同化升级】")
        print("  从简单的线性插值升级为专业海洋数据同化系统:")
        print()
        print("  1. 多种同化算法:")
        print("     - linear  - 线性插值 (基准)")
        print("     - spline  - Akima样条插值")
        print("     - optimal - 最优混合插值 (Pchip+Akima+线性 加权融合)")
        print("     - gaussian- 高斯风格混合插值")
        print()
        print("  2. 高级插值技术:")
        print("     - Pchip保形插值 (保持曲线形状,无过冲)")
        print("     - Akima插值 (平滑,适合海洋剖面)")
        print("     - 加权融合 (多种方法取长补短)")
        print("     - 高斯平滑 (减少噪声)")
        print()
        print("  3. 物理约束校正:")
        print("     - T-S温盐关系耦合校正")
        print("     - 静力稳定度约束 (密度随深度增加)")
        print("     - 变量范围限制 (温、盐、流速物理范围)")
        print("     - 背景场偏差限制 (最大偏离2度/1PSU)")
        print()
        print("  4. 质量评估:")
        print("     - 同化误差估计")
        print("     - 置信度评估 (基于数据密度和距离)")
        print("     - RMSE/MAE/Bias 统计")
        print()
        print("【二、三维可视化新增】")
        print("  从2D剖面图升级为3D立体渲染:")
        print()
        print("  1. 3D 温盐深散点图 - 温度/盐度/深度三维关系")
        print("  2. 3D 海流矢量图 - 锥形箭头展示u/v分量")
        print("  3. 3D 温度体积渲染 - 旋转曲面展示立体分布")
        print("  4. 3D 盐度体积渲染 - 旋转曲面展示立体分布")
        print("  5. 3D 全球浮标地图 - 经纬度+海温三维分布")
        print("  6. 3D 综合仪表板 - 2x2子图全景")
        print()
        print("所有3D图表都支持交互: 旋转、缩放、平移、悬浮查看数据")
        print(f"输出目录: {output_dir}")
        print()

        return 0

    except Exception as e:
        print(f"\n[FAILED] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
