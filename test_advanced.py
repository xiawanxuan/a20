import sys
sys.path.insert(0, '.')

import numpy as np
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController
from visualization import OceanVisualizer
from alerts import AlertManager
from main import DataExporter


def test_advanced_assimilation():
    print("=" * 60)
    print("Testing Advanced Data Assimilation")
    print("=" * 60)

    fetcher = BuoyDataFetcher(use_simulation=True)
    raw_data = fetcher.fetch_realtime_data('46001')

    df_corrupted = raw_data.copy()
    n_corrupt = 15
    corrupt_indices = np.random.choice(len(df_corrupted), n_corrupt, replace=False)
    df_corrupted.loc[corrupt_indices, 'temperature'] += np.random.uniform(-5, 5, n_corrupt)
    df_corrupted.loc[corrupt_indices, 'salinity'] += np.random.uniform(-3, 3, n_corrupt)
    df_corrupted.loc[corrupt_indices, 'current_speed'] += np.random.uniform(-0.5, 1.5, n_corrupt)

    methods = ['linear', 'spline', 'optimal']
    results = {}

    for method in methods:
        print(f"\n--- Testing {method} assimilation ---")
        qc = QualityController(assimilation_method=method)
        qc_data = qc.run_full_qc(df_corrupted)

        good_mask = ~qc_data['data_assimilated']
        if good_mask.sum() > 0 and qc_data['data_assimilated'].sum() > 0:
            temp_orig = raw_data.loc[qc_data['data_assimilated'], 'temperature'].values
            temp_assim = qc_data.loc[qc_data['data_assimilated'], 'temperature_assimilated'].values
            temp_rmse = np.sqrt(np.mean((temp_orig - temp_assim) ** 2))

            sal_orig = raw_data.loc[qc_data['data_assimilated'], 'salinity'].values
            sal_assim = qc_data.loc[qc_data['data_assimilated'], 'salinity_assimilated'].values
            sal_rmse = np.sqrt(np.mean((sal_orig - sal_assim) ** 2))

            results[method] = {
                'temp_rmse': temp_rmse,
                'sal_rmse': sal_rmse,
                'assimilated_count': qc_data['data_assimilated'].sum(),
            }

            print(f"  Assimilated points: {qc_data['data_assimilated'].sum()}")
            print(f"  Temperature RMSE: {temp_rmse:.4f} °C")
            print(f"  Salinity RMSE: {sal_rmse:.4f} PSU")
            if 'assimilation_confidence' in qc_data.columns:
                print(f"  Mean confidence: {qc_data['assimilation_confidence'].mean():.4f}")
        else:
            print(f"  Not enough data for comparison")
            results[method] = None

    print("\n" + "=" * 60)
    print("Assimilation Method Comparison")
    print("=" * 60)
    if all(v is not None for v in results.values()):
        print(f"{'Method':<15} {'Temp RMSE':<15} {'Sal RMSE':<15}")
        print("-" * 45)
        for method, res in results.items():
            print(f"{method:<15} {res['temp_rmse']:<15.4f} {res['sal_rmse']:<15.4f}")

        best_method = min(results.keys(), key=lambda m: results[m]['temp_rmse'])
        print(f"\nBest temperature method: {best_method}")
    else:
        print("All methods completed (no comparison available)")

    print("\n[OK] Advanced assimilation test passed")
    print()
    return results


def test_3d_visualization():
    print("=" * 60)
    print("Testing 3D Visualization")
    print("=" * 60)

    fetcher = BuoyDataFetcher(use_simulation=True)
    data = fetcher.fetch_realtime_data('46001')

    qc = QualityController(assimilation_method='optimal')
    qc_data = qc.run_full_qc(data)

    viz = OceanVisualizer()

    print("\n1. 3D T-S Scatter Plot...")
    fig_3d_ts = viz.create_3d_ts_scatter(qc_data)
    print(f"   Created: {type(fig_3d_ts).__name__}")

    print("\n2. 3D Current Vectors...")
    fig_3d_current = viz.create_3d_current_vectors(qc_data)
    print(f"   Created: {type(fig_3d_current).__name__}")

    print("\n3. 3D Temperature Volume...")
    fig_3d_temp = viz.create_3d_temperature_volume(qc_data)
    print(f"   Created: {type(fig_3d_temp).__name__}")

    print("\n4. 3D Salinity Volume...")
    fig_3d_sal = viz.create_3d_salinity_volume(qc_data)
    print(f"   Created: {type(fig_3d_sal).__name__}")

    print("\n5. 3D Multi-Buoy Map...")
    buoys = fetcher.fetch_buoy_list()
    surface_data = fetcher.fetch_surface_data()
    fig_3d_map = viz.create_3d_multi_buoy_map(buoys, surface_data)
    print(f"   Created: {type(fig_3d_map).__name__}")

    print("\n6. 3D Dashboard...")
    fig_3d_dashboard = viz.create_3d_dashboard(qc_data)
    print(f"   Created: {type(fig_3d_dashboard).__name__}")

    print("\n[OK] All 3D visualizations created successfully")
    print()

    return {
        '3d_ts_scatter': fig_3d_ts,
        '3d_current': fig_3d_current,
        '3d_temp_volume': fig_3d_temp,
        '3d_sal_volume': fig_3d_sal,
        '3d_map': fig_3d_map,
        '3d_dashboard': fig_3d_dashboard,
    }


def test_export_with_3d():
    print("=" * 60)
    print("Testing Export with 3D Visualizations")
    print("=" * 60)

    fetcher = BuoyDataFetcher(use_simulation=True)
    data = fetcher.fetch_realtime_data('46001')

    qc = QualityController(assimilation_method='optimal')
    qc_data = qc.run_full_qc(data)
    qc_summary = qc.get_quality_summary(qc_data)

    exporter = DataExporter()

    viz = OceanVisualizer()
    fig_3d = viz.create_3d_dashboard(qc_data)
    path = exporter.export_figure(fig_3d, 'test_3d_dashboard.html')
    print(f"3D Dashboard exported: {path}")

    print("\nExporting full report with 3D visualizations...")
    results = exporter.export_full_report(qc_data, qc_summary, None, buoy_id='46001')
    print(f"Total files exported: {len(results)}")
    print("\nFiles:")
    for key, filepath in results.items():
        print(f"  - {key}: {os.path.basename(filepath)}")

    print("\n[OK] Export test passed")
    print()


import os

def main():
    print("\n" + "=" * 60)
    print("Advanced Features Test - Assimilation & 3D Visualization")
    print("=" * 60 + "\n")

    try:
        test_advanced_assimilation()
        test_3d_visualization()
        test_export_with_3d()

        print("=" * 60)
        print("[SUCCESS] All advanced features tested successfully!")
        print("=" * 60)
        print("\nKey improvements:")
        print("  1. Data Assimilation:")
        print("     - Spline interpolation (smoother curves)")
        print("     - Optimal interpolation (with background field)")
        print("     - T-S physical constraint (thermodynamic consistency)")
        print("     - Gaussian process/RBF interpolation")
        print("     - Confidence estimation")
        print()
        print("  2. 3D Visualization:")
        print("     - 3D T-S scatter plot (with depth axis)")
        print("     - 3D current velocity vectors (cones)")
        print("     - 3D temperature volume rendering")
        print("     - 3D salinity volume rendering")
        print("     - 3D multi-buoy map")
        print("     - 3D comprehensive dashboard")
        print()
        return 0

    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
