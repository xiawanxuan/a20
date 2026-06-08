import sys
sys.path.insert(0, '.')

from data_acquisition import BuoyDataFetcher
from quality_control import QualityController
from visualization import OceanVisualizer
from alerts import AlertManager
from main import DataExporter

def test_data_acquisition():
    print("=" * 50)
    print("Testing Data Acquisition Module...")
    print("=" * 50)
    
    fetcher = BuoyDataFetcher(use_simulation=True)
    buoys = fetcher.fetch_buoy_list()
    print(f"Available buoys: {len(buoys)}")
    
    data = fetcher.fetch_realtime_data('46001')
    print(f"Profile data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Depth range: {data['depth'].min():.1f} - {data['depth'].max():.1f} m")
    print(f"Temperature range: {data['temperature'].min():.2f} - {data['temperature'].max():.2f} °C")
    
    surface = fetcher.fetch_surface_data('46001')
    print(f"Surface temperature: {surface['temperature']:.2f} °C")
    print(f"Surface salinity: {surface['salinity']:.2f} PSU")
    
    print("[OK] Data acquisition module")
    print()
    return data

def test_quality_control(data):
    print("=" * 50)
    print("Testing Quality Control Module...")
    print("=" * 50)
    
    qc = QualityController()
    qc_data = qc.run_full_qc(data)
    print(f"QC data shape: {qc_data.shape}")
    print(f"Quality flags: {qc_data['quality_flag'].value_counts().to_dict()}")
    
    summary = qc.get_quality_summary(qc_data)
    print(f"Good percentage: {summary['good_percentage']}%")
    print(f"Assimilated count: {summary['assimilated_count']}")
    
    anomalies = qc.get_anomalies(qc_data)
    print(f"Anomalies found: {len(anomalies)}")
    
    print("[OK] Quality control module")
    print()
    return qc_data, summary

def test_visualization(qc_data):
    print("=" * 50)
    print("Testing Visualization Module...")
    print("=" * 50)
    
    viz = OceanVisualizer()
    
    temp_fig = viz.create_temperature_profile(qc_data)
    print(f"Temperature profile figure created: {type(temp_fig).__name__}")
    
    sal_fig = viz.create_salinity_profile(qc_data)
    print(f"Salinity profile figure created: {type(sal_fig).__name__}")
    
    ts_fig = viz.create_ts_diagram(qc_data)
    print(f"T-S diagram figure created: {type(ts_fig).__name__}")
    
    current_fig = viz.create_current_vector_plot(qc_data)
    print(f"Current vector plot figure created: {type(current_fig).__name__}")
    
    dashboard = viz.create_dashboard(qc_data)
    print(f"Dashboard figure created: {type(dashboard).__name__}")
    
    qc_fig = viz.create_quality_control_plot(qc_data)
    print(f"QC plot figure created: {type(qc_fig).__name__}")
    
    assim_fig = viz.create_data_assimilation_plot(qc_data)
    print(f"Data assimilation plot figure created: {type(assim_fig).__name__}")
    
    print("[OK] Visualization module")
    print()
    return dashboard

def test_alerts(qc_data, qc_summary):
    print("=" * 50)
    print("Testing Alert Manager Module...")
    print("=" * 50)
    
    alert_mgr = AlertManager()
    alerts = alert_mgr.check_alerts(qc_data, qc_summary)
    print(f"Alerts generated: {len(alerts)}")
    
    summary = alert_mgr.get_alert_summary()
    print(f"Total alerts in history: {summary['total_alerts']}")
    
    print("[OK] Alert manager module")
    print()
    return alerts

def test_export(qc_data, qc_summary, dashboard):
    print("=" * 50)
    print("Testing Data Export Module...")
    print("=" * 50)
    
    exporter = DataExporter()
    
    csv_path = exporter.export_to_csv(qc_data, 'test_data.csv')
    print(f"CSV exported: {csv_path}")
    
    json_path = exporter.export_to_json(qc_data, 'test_data.json')
    print(f"JSON exported: {json_path}")
    
    html_path = exporter.export_figure(dashboard, 'test_dashboard.html')
    print(f"Dashboard HTML exported: {html_path}")
    
    qc_report = exporter.export_quality_report(qc_summary, 'test_quality_report.json')
    print(f"Quality report exported: {qc_report}")
    
    print("[OK] Data export module")
    print()

def main():
    print("\n" + "=" * 60)
    print("Global Ocean Buoy System - Module Integration Test")
    print("=" * 60 + "\n")
    
    try:
        raw_data = test_data_acquisition()
        qc_data, qc_summary = test_quality_control(raw_data)
        dashboard = test_visualization(qc_data)
        test_alerts(qc_data, qc_summary)
        test_export(qc_data, qc_summary, dashboard)
        
        print("=" * 60)
        print("[SUCCESS] ALL MODULES TESTED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files are located in the 'output' directory.")
        return 0
        
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
