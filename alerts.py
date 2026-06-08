import os
import time
import json
from datetime import datetime
from collections import defaultdict
from config import ALERT_CONFIG, QUALITY_CONTROL


class AlertManager:
    def __init__(self):
        self.config = ALERT_CONFIG
        self.qc_rules = QUALITY_CONTROL
        self.alert_history = []
        self.last_alert_time = defaultdict(float)
        self.alert_counts = defaultdict(int)

        alert_file = self.config['alert_file']
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)

    def check_alerts(self, data, quality_summary=None):
        """Check data for anomalies and generate alerts."""
        alerts = []

        if isinstance(data, dict):
            for buoy_id, df in data.items():
                buoy_alerts = self._check_buoy_data(buoy_id, df)
                alerts.extend(buoy_alerts)
        else:
            buoy_id = data['buoy_id'].iloc[0] if 'buoy_id' in data.columns else 'unknown'
            buoy_alerts = self._check_buoy_data(buoy_id, data)
            alerts.extend(buoy_alerts)

        if quality_summary:
            qc_alerts = self._check_quality_summary(quality_summary)
            alerts.extend(qc_alerts)

        filtered_alerts = self._apply_cooldown(alerts)
        self._process_alerts(filtered_alerts)

        return filtered_alerts

    def _check_buoy_data(self, buoy_id, df):
        alerts = []

        if 'quality_flag' not in df.columns:
            return alerts

        bad_count = (df['quality_flag'] == 'bad').sum()
        suspect_count = (df['quality_flag'] == 'suspect').sum()
        missing_count = (df['quality_flag'] == 'missing').sum()
        total = len(df)

        if total == 0:
            return alerts

        bad_pct = bad_count / total * 100
        suspect_pct = suspect_count / total * 100

        if bad_pct > 10:
            alerts.append({
                'buoy_id': buoy_id,
                'level': 'critical',
                'type': 'high_bad_data_ratio',
                'message': f'Buoy {buoy_id}: {bad_pct:.1f}% bad data points detected',
                'value': bad_pct,
                'threshold': 10.0,
                'timestamp': datetime.now(),
            })

        if suspect_pct > 20:
            alerts.append({
                'buoy_id': buoy_id,
                'level': 'warning',
                'type': 'high_suspect_data_ratio',
                'message': f'Buoy {buoy_id}: {suspect_pct:.1f}% suspect data points detected',
                'value': suspect_pct,
                'threshold': 20.0,
                'timestamp': datetime.now(),
            })

        temp_anomalies = df[df['qc_details'].str.contains('temperature', na=False)]
        sal_anomalies = df[df['qc_details'].str.contains('salinity', na=False)]

        if len(temp_anomalies) > total * 0.05:
            alerts.append({
                'buoy_id': buoy_id,
                'level': 'warning',
                'type': 'temperature_anomalies',
                'message': f'Buoy {buoy_id}: Multiple temperature anomalies detected ({len(temp_anomalies)} points)',
                'value': len(temp_anomalies),
                'threshold': total * 0.05,
                'timestamp': datetime.now(),
            })

        if len(sal_anomalies) > total * 0.05:
            alerts.append({
                'buoy_id': buoy_id,
                'level': 'warning',
                'type': 'salinity_anomalies',
                'message': f'Buoy {buoy_id}: Multiple salinity anomalies detected ({len(sal_anomalies)} points)',
                'value': len(sal_anomalies),
                'threshold': total * 0.05,
                'timestamp': datetime.now(),
            })

        if 'current_speed' in df.columns:
            max_speed = df['current_speed'].max()
            if max_speed > self.qc_rules['current_speed_max'] * 0.8:
                alerts.append({
                    'buoy_id': buoy_id,
                    'level': 'warning',
                    'type': 'high_current_speed',
                    'message': f'Buoy {buoy_id}: High current speed detected ({max_speed:.2f} m/s)',
                    'value': max_speed,
                    'threshold': self.qc_rules['current_speed_max'] * 0.8,
                    'timestamp': datetime.now(),
                })

        return alerts

    def _check_quality_summary(self, quality_summary):
        alerts = []

        if isinstance(quality_summary, dict):
            if 'total_points' in quality_summary:
                if quality_summary.get('bad_percentage', 0) > 15:
                    alerts.append({
                        'buoy_id': 'unknown',
                        'level': 'critical',
                        'type': 'quality_degradation',
                        'message': f'Data quality degraded to {quality_summary["good_percentage"]:.1f}% good',
                        'value': quality_summary['good_percentage'],
                        'threshold': 85.0,
                        'timestamp': datetime.now(),
                    })
            else:
                for buoy_id, summary in quality_summary.items():
                    if isinstance(summary, dict) and summary.get('bad_percentage', 0) > 15:
                        alerts.append({
                            'buoy_id': buoy_id,
                            'level': 'critical',
                            'type': 'quality_degradation',
                            'message': f'Buoy {buoy_id}: Data quality degraded to {summary["good_percentage"]:.1f}% good',
                            'value': summary['good_percentage'],
                            'threshold': 85.0,
                            'timestamp': datetime.now(),
                        })

        return alerts

    def _apply_cooldown(self, alerts):
        filtered = []
        now = time.time()
        cooldown = self.config['alert_cooldown']

        for alert in alerts:
            alert_key = f"{alert['buoy_id']}_{alert['type']}"
            if now - self.last_alert_time[alert_key] > cooldown:
                filtered.append(alert)
                self.last_alert_time[alert_key] = now
                self.alert_counts[alert_key] += 1

        return filtered

    def _process_alerts(self, alerts):
        for alert in alerts:
            self.alert_history.append(alert)

            if self.config['enable_console']:
                self._print_alert(alert)

            if self.config['alert_file']:
                self._log_alert_to_file(alert)

    def _print_alert(self, alert):
        level_colors = {
            'critical': '\033[91m',
            'warning': '\033[93m',
            'info': '\033[92m',
        }
        reset = '\033[0m'
        color = level_colors.get(alert['level'], '')
        print(f"{color}[{alert['level'].upper()}] {alert['message']}{reset}")

    def _log_alert_to_file(self, alert):
        log_entry = {
            'timestamp': alert['timestamp'].isoformat() if hasattr(alert['timestamp'], 'isoformat') else str(alert['timestamp']),
            'buoy_id': alert['buoy_id'],
            'level': alert['level'],
            'type': alert['type'],
            'message': alert['message'],
            'value': alert.get('value'),
            'threshold': alert.get('threshold'),
        }

        with open(self.config['alert_file'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_alert_history(self, buoy_id=None, level=None, limit=100):
        history = self.alert_history

        if buoy_id:
            history = [a for a in history if a['buoy_id'] == buoy_id]

        if level:
            history = [a for a in history if a['level'] == level]

        return history[-limit:]

    def get_alert_summary(self):
        total = len(self.alert_history)
        by_level = defaultdict(int)
        by_buoy = defaultdict(int)
        by_type = defaultdict(int)

        for alert in self.alert_history:
            by_level[alert['level']] += 1
            by_buoy[alert['buoy_id']] += 1
            by_type[alert['type']] += 1

        return {
            'total_alerts': total,
            'by_level': dict(by_level),
            'by_buoy': dict(by_buoy),
            'by_type': dict(by_type),
            'repeated_alerts': {k: v for k, v in self.alert_counts.items() if v > 1},
        }

    def clear_alerts(self):
        self.alert_history = []
        self.alert_counts.clear()
        self.last_alert_time.clear()

    def send_email_alert(self, alert, recipients=None):
        if not self.config['enable_email']:
            return False

        print(f"[EMAIL ALERT] Would send email to {recipients}: {alert['message']}")
        return True

    def create_alert_summary_plot(self):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            summary = self.get_alert_summary()

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Alerts by Level', 'Alerts by Type'),
                specs=[[{'type': 'pie'}, {'type': 'bar'}]],
            )

            levels = list(summary['by_level'].keys())
            level_counts = list(summary['by_level'].values())
            level_colors = {'critical': 'red', 'warning': 'orange', 'info': 'green'}
            colors = [level_colors.get(l, 'gray') for l in levels]

            fig.add_trace(go.Pie(
                labels=levels,
                values=level_counts,
                marker=dict(colors=colors),
                name='By Level',
            ), row=1, col=1)

            types = list(summary['by_type'].keys())
            type_counts = list(summary['by_type'].values())

            fig.add_trace(go.Bar(
                x=types,
                y=type_counts,
                name='By Type',
                marker_color='steelblue',
            ), row=1, col=2)

            fig.update_layout(
                title='Alert Summary',
                template='plotly_dark',
                height=400,
                showlegend=True,
            )

            return fig
        except ImportError:
            return None
