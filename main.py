import os
import sys
import time
import json
import pandas as pd
from datetime import datetime
from config import OUTPUT_DIR, DATA_DIR, BUOY_CONFIG
from data_acquisition import BuoyDataFetcher
from quality_control import QualityController
from visualization import OceanVisualizer
from alerts import AlertManager


class DataExporter:
    def __init__(self):
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def export_to_csv(self, data, filename):
        filepath = os.path.join(self.output_dir, filename)
        if isinstance(data, dict):
            all_dfs = []
            for buoy_id, df in data.items():
                all_dfs.append(df)
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(filepath, index=False)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        return filepath

    def export_to_json(self, data, filename):
        filepath = os.path.join(self.output_dir, filename)
        if isinstance(data, dict):
            result = {}
            for buoy_id, df in data.items():
                result[buoy_id] = df.to_dict('records')
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        elif isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient='records', indent=2, date_format='iso')
        return filepath

    def export_quality_report(self, quality_summary, filename):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(quality_summary, f, indent=2, default=str)
        return filepath

    def export_figure(self, fig, filename, fmt='html'):
        filepath = os.path.join(self.output_dir, filename)
        if fmt == 'html':
            fig.write_html(filepath)
        elif fmt == 'png':
            fig.write_image(filepath)
        elif fmt == 'svg':
            fig.write_image(filepath)
        return filepath

    def export_full_report(self, data, quality_summary, alerts_summary, buoy_id=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"report_{buoy_id or 'all'}_{timestamp}"

        results = {}

        if isinstance(data, dict) and buoy_id:
            df = data[buoy_id]
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            df = None

        if df is not None:
            csv_path = self.export_to_csv(df, f"{report_name}.csv")
            results['data_csv'] = csv_path

            visualizer = OceanVisualizer()

            temp_fig = visualizer.create_temperature_profile(df)
            temp_html = self.export_figure(temp_fig, f"{report_name}_temp_profile.html")
            results['temp_profile'] = temp_html

            sal_fig = visualizer.create_salinity_profile(df)
            sal_html = self.export_figure(sal_fig, f"{report_name}_salinity_profile.html")
            results['salinity_profile'] = sal_html

            ts_fig = visualizer.create_ts_diagram(df)
            ts_html = self.export_figure(ts_fig, f"{report_name}_ts_diagram.html")
            results['ts_diagram'] = ts_html

            current_fig = visualizer.create_current_vector_plot(df)
            current_html = self.export_figure(current_fig, f"{report_name}_current_profile.html")
            results['current_profile'] = current_html

            dashboard_fig = visualizer.create_dashboard(df)
            dashboard_html = self.export_figure(dashboard_fig, f"{report_name}_dashboard.html")
            results['dashboard'] = dashboard_html

            if 'quality_flag' in df.columns:
                qc_fig = visualizer.create_quality_control_plot(df)
                qc_html = self.export_figure(qc_fig, f"{report_name}_qc_results.html")
                results['qc_plot'] = qc_html

            if 'temperature_assimilated' in df.columns:
                assim_fig = visualizer.create_data_assimilation_plot(df)
                assim_html = self.export_figure(assim_fig, f"{report_name}_assimilation.html")
                results['assimilation_plot'] = assim_html

            fig_3d_ts = visualizer.create_3d_ts_scatter(df)
            html_3d_ts = self.export_figure(fig_3d_ts, f"{report_name}_3d_ts_scatter.html")
            results['3d_ts_scatter'] = html_3d_ts

            fig_3d_temp = visualizer.create_3d_temperature_volume(df)
            html_3d_temp = self.export_figure(fig_3d_temp, f"{report_name}_3d_temp_volume.html")
            results['3d_temp_volume'] = html_3d_temp

            fig_3d_sal = visualizer.create_3d_salinity_volume(df)
            html_3d_sal = self.export_figure(fig_3d_sal, f"{report_name}_3d_salinity_volume.html")
            results['3d_salinity_volume'] = html_3d_sal

            fig_3d_current = visualizer.create_3d_current_vectors(df)
            html_3d_current = self.export_figure(fig_3d_current, f"{report_name}_3d_current_vectors.html")
            results['3d_current_vectors'] = html_3d_current

            fig_3d_dashboard = visualizer.create_3d_dashboard(df)
            html_3d_dashboard = self.export_figure(fig_3d_dashboard, f"{report_name}_3d_dashboard.html")
            results['3d_dashboard'] = html_3d_dashboard

        if quality_summary:
            qc_path = self.export_quality_report(quality_summary, f"{report_name}_quality_report.json")
            results['quality_report'] = qc_path

        print(f"Report exported to {self.output_dir}")
        print(f"Files generated: {len(results)}")

        return results


class OceanBuoySystem:
    def __init__(self, use_simulation=True, assimilation_method='optimal'):
        self.fetcher = BuoyDataFetcher(use_simulation=use_simulation)
        self.quality_controller = QualityController(assimilation_method=assimilation_method)
        self.visualizer = OceanVisualizer()
        self.alert_manager = AlertManager()
        self.exporter = DataExporter()

        self.current_data = None
        self.current_qc_data = None
        self.quality_summary = None
        self.mode = 'offline'
        self.assimilation_method = assimilation_method

    def run_offline_analysis(self, filepath=None, buoy_id=None):
        print("=" * 60)
        print("Running Offline Data Analysis")
        print("=" * 60)

        if filepath and os.path.exists(filepath):
            print(f"Loading data from: {filepath}")
            raw_data = self.fetcher.load_offline_data(filepath)
        else:
            print("Generating sample data for demonstration...")
            raw_data = self.fetcher.fetch_realtime_data(buoy_id)

        print("Applying quality control and data assimilation...")
        self.current_qc_data = self.quality_controller.run_full_qc(raw_data)
        self.quality_summary = self.quality_controller.get_quality_summary(self.current_qc_data)

        print("\nQuality Summary:")
        if isinstance(self.quality_summary, dict):
            if 'total_points' in self.quality_summary:
                self._print_quality_summary(self.quality_summary)
            else:
                for bid, summary in self.quality_summary.items():
                    print(f"\n  Buoy {bid}:")
                    self._print_quality_summary(summary, indent=4)

        print("\nChecking for alerts...")
        alerts = self.alert_manager.check_alerts(self.current_qc_data, self.quality_summary)
        if alerts:
            print(f"Generated {len(alerts)} alert(s)")
        else:
            print("No alerts triggered")

        print("\nGenerating visualizations...")
        self._show_dashboard_preview()

        print("\nExporting analysis results...")
        if buoy_id or isinstance(self.current_qc_data, dict):
            target_buoy = buoy_id or list(self.current_qc_data.keys())[0]
            results = self.exporter.export_full_report(
                self.current_qc_data,
                self.quality_summary,
                self.alert_manager.get_alert_summary(),
                buoy_id=target_buoy
            )
            print(f"Report exported with {len(results)} files")
        else:
            results = self.exporter.export_full_report(
                self.current_qc_data,
                self.quality_summary,
                self.alert_manager.get_alert_summary()
            )
            print(f"Report exported with {len(results)} files")

        return {
            'data': self.current_qc_data,
            'quality_summary': self.quality_summary,
            'alerts': alerts,
            'alert_summary': self.alert_manager.get_alert_summary(),
        }

    def run_realtime_demo(self, duration_seconds=60, update_interval=10):
        print("=" * 60)
        print("Running Real-Time Data Demonstration")
        print(f"Duration: {duration_seconds}s, Update interval: {update_interval}s")
        print("=" * 60)

        self.mode = 'realtime'
        start_time = time.time()
        update_count = 0

        while time.time() - start_time < duration_seconds:
            update_count += 1
            print(f"\n--- Update #{update_count} ({datetime.now().strftime('%H:%M:%S')}) ---")

            print("Fetching real-time data...")
            raw_data = self.fetcher.fetch_realtime_data()
            self.current_data = raw_data

            print("Applying QC...")
            self.current_qc_data = self.quality_controller.run_full_qc(raw_data)
            self.quality_summary = self.quality_controller.get_quality_summary(self.current_qc_data)

            print("Checking alerts...")
            alerts = self.alert_manager.check_alerts(self.current_qc_data, self.quality_summary)

            first_buoy = list(self.current_qc_data.keys())[0]
            summary = self.quality_summary.get(first_buoy, {})
            good_pct = summary.get('good_percentage', 0)
            print(f"  Buoy {first_buoy}: {good_pct:.1f}% good data")
            if alerts:
                print(f"  Active alerts: {len(alerts)}")

            if update_count < duration_seconds / update_interval:
                print(f"Next update in {update_interval}s...")
                time.sleep(update_interval)

        print("\n" + "=" * 60)
        print("Real-time demonstration complete")
        print("=" * 60)

        alert_summary = self.alert_manager.get_alert_summary()
        print(f"\nTotal alerts generated: {alert_summary['total_alerts']}")

        self._export_realtime_summary()

        return {
            'updates': update_count,
            'alert_summary': alert_summary,
            'final_data': self.current_qc_data,
        }

    def _print_quality_summary(self, summary, indent=0):
        prefix = " " * indent
        print(f"{prefix}Total points: {summary['total_points']}")
        print(f"{prefix}Good: {summary['good_count']} ({summary['good_percentage']}%)")
        print(f"{prefix}Suspect: {summary['suspect_count']} ({summary['suspect_percentage']}%)")
        print(f"{prefix}Bad: {summary['bad_count']} ({summary['bad_percentage']}%)")
        if 'assimilated_count' in summary:
            print(f"{prefix}Assimilated: {summary['assimilated_count']}")

    def _show_dashboard_preview(self):
        if isinstance(self.current_qc_data, dict):
            first_buoy = list(self.current_qc_data.keys())[0]
            df = self.current_qc_data[first_buoy]
        else:
            df = self.current_qc_data

        print(f"  Dashboard generated for buoy data ({len(df)} depth levels)")
        print(f"  Temperature range: {df['temperature'].min():.2f} - {df['temperature'].max():.2f} °C")
        print(f"  Salinity range: {df['salinity'].min():.2f} - {df['salinity'].max():.2f} PSU")
        print(f"  Max depth: {df['depth'].max():.0f} m")

    def _export_realtime_summary(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.current_qc_data:
            data_path = self.exporter.export_to_csv(
                self.current_qc_data,
                f"realtime_data_{timestamp}.csv"
            )
            print(f"Final data exported: {data_path}")

        alert_summary = self.alert_manager.get_alert_summary()
        summary_path = self.exporter.export_quality_report(
            alert_summary,
            f"alert_summary_{timestamp}.json"
        )
        print(f"Alert summary exported: {summary_path}")

        if self.current_qc_data:
            visualizer = OceanVisualizer()
            first_buoy = list(self.current_qc_data.keys())[0]
            fig = visualizer.create_dashboard(self.current_qc_data, first_buoy)
            dashboard_path = self.exporter.export_figure(
                fig,
                f"realtime_dashboard_{timestamp}.html"
            )
            print(f"Dashboard exported: {dashboard_path}")

    def list_buoys(self):
        return self.fetcher.fetch_buoy_list()

    def get_buoy_data(self, buoy_id):
        if buoy_id in (self.current_qc_data or {}):
            return self.current_qc_data[buoy_id]
        else:
            raw_data = self.fetcher.fetch_realtime_data(buoy_id)
            return self.quality_controller.run_full_qc(raw_data)

    def export_buoy_report(self, buoy_id):
        data = self.get_buoy_data(buoy_id)
        qc_summary = self.quality_controller.get_quality_summary(data)
        return self.exporter.export_full_report(data, qc_summary, None, buoy_id=buoy_id)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Global Ocean Buoy Data Assimilation and Visualization System'
    )
    parser.add_argument(
        '--mode', choices=['offline', 'realtime', 'demo'],
        default='demo',
        help='Operation mode: offline, realtime, or demo (default: demo)'
    )
    parser.add_argument(
        '--file', type=str, default=None,
        help='Path to offline data file (for offline mode)'
    )
    parser.add_argument(
        '--buoy', type=str, default=None,
        help='Buoy ID to analyze'
    )
    parser.add_argument(
        '--duration', type=int, default=60,
        help='Duration of real-time demo in seconds (default: 60)'
    )
    parser.add_argument(
        '--interval', type=int, default=10,
        help='Update interval for real-time mode in seconds (default: 10)'
    )
    parser.add_argument(
        '--list-buoys', action='store_true',
        help='List available buoys and exit'
    )

    args = parser.parse_args()

    system = OceanBuoySystem(use_simulation=True)

    if args.list_buoys:
        buoys = system.list_buoys()
        print("\nAvailable Buoys:")
        print("-" * 60)
        for buoy in buoys:
            print(f"  {buoy['id']}: {buoy['name']} (Lat: {buoy['lat']}, Lon: {buoy['lon']})")
        return

    if args.mode == 'offline':
        system.run_offline_analysis(filepath=args.file, buoy_id=args.buoy)
    elif args.mode == 'realtime':
        system.run_realtime_demo(
            duration_seconds=args.duration,
            update_interval=args.interval
        )
    else:
        print("Running demo mode...\n")
        buoys = system.list_buoys()
        print(f"Network: {len(buoys)} buoys available")
        print("\nRunning offline analysis demo...")
        system.run_offline_analysis(buoy_id=args.buoy)
        print("\n" + "=" * 60)
        print("Demo complete! Check output directory for results.")
        print("=" * 60)


if __name__ == '__main__':
    main()
