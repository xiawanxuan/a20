import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import VISUALIZATION_CONFIG


class OceanVisualizer:
    def __init__(self):
        self.template = VISUALIZATION_CONFIG['default_template']
        self.colorscale_temp = VISUALIZATION_CONFIG['colorscale_temp']
        self.colorscale_salinity = VISUALIZATION_CONFIG['colorscale_salinity']

    def create_temperature_profile(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['temperature'],
            y=df['depth'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='red', width=2),
            marker=dict(size=4),
        ))

        if 'temperature_assimilated' in df.columns:
            assimilated_mask = df['data_assimilated'] if 'data_assimilated' in df.columns else pd.Series([False] * len(df))
            fig.add_trace(go.Scatter(
                x=df.loc[assimilated_mask, 'temperature_assimilated'],
                y=df.loc[assimilated_mask, 'depth'],
                mode='markers',
                name='Assimilated',
                marker=dict(color='orange', size=8, symbol='x'),
            ))

        fig.update_layout(
            title=f'Temperature Profile - {buoy_name}',
            xaxis_title='Temperature (°C)',
            yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            template=self.template,
            height=600,
        )

        return fig

    def create_salinity_profile(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['salinity'],
            y=df['depth'],
            mode='lines+markers',
            name='Salinity',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
        ))

        fig.update_layout(
            title=f'Salinity Profile - {buoy_name}',
            xaxis_title='Salinity (PSU)',
            yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            template=self.template,
            height=600,
        )

        return fig

    def create_ts_diagram(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[sal_col],
            y=df[temp_col],
            mode='markers',
            marker=dict(
                size=8,
                color=df['depth'],
                colorscale='Viridis',
                colorbar=dict(title='Depth (m)'),
            ),
            text=[f'Depth: {d:.1f}m' for d in df['depth']],
            hoverinfo='text+x+y',
        ))

        fig.update_layout(
            title=f'T-S Diagram - {buoy_name}',
            xaxis_title='Salinity (PSU)',
            yaxis_title='Temperature (°C)',
            template=self.template,
            height=600,
        )

        return fig

    def create_current_vector_plot(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Current Speed Profile', 'Current Direction Profile'),
            shared_yaxes=True,
        )

        fig.add_trace(go.Scatter(
            x=df['current_speed'],
            y=df['depth'],
            mode='lines+markers',
            name='Current Speed',
            line=dict(color='green', width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['current_direction'],
            y=df['depth'],
            mode='lines+markers',
            name='Current Direction',
            line=dict(color='purple', width=2),
        ), row=1, col=2)

        fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
        fig.update_xaxes(title_text='Speed (m/s)', row=1, col=1)
        fig.update_xaxes(title_text='Direction (°)', row=1, col=2)

        fig.update_layout(
            title=f'Current Velocity Profile - {buoy_name}',
            template=self.template,
            height=600,
            showlegend=False,
        )

        return fig

    def create_current_rose(self, data, buoy_id=None, depth_range=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        if depth_range:
            df = df[(df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1])]

        directions = df['current_direction'].values
        speeds = df['current_speed'].values

        dir_bins = np.arange(0, 360, 30)
        speed_bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        speed_labels = ['0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0']

        counts = np.zeros((len(dir_bins), len(speed_bins) - 1))

        for d, s in zip(directions, speeds):
            dir_idx = int(d // 30) % 12
            for i in range(len(speed_bins) - 1):
                if speed_bins[i] <= s < speed_bins[i + 1]:
                    counts[dir_idx, i] += 1
                    break

        colors = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58']

        fig = go.Figure()

        for i in range(len(speed_bins) - 1):
            fig.add_trace(go.Barpolar(
                r=counts[:, i],
                theta=dir_bins,
                width=25,
                name=speed_labels[i],
                marker_color=colors[i % len(colors)],
            ))

        fig.update_layout(
            title=f'Current Rose - {buoy_name}',
            template=self.template,
            polar=dict(
                radialaxis=dict(title='Count'),
                angularaxis=dict(direction='clockwise', rotation=90),
            ),
            height=600,
        )

        return fig

    def create_combined_profile(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Temperature', 'Salinity', 'Current Speed'),
            shared_yaxes=True,
        )

        fig.add_trace(go.Scatter(
            x=df['temperature'],
            y=df['depth'],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='red'),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['salinity'],
            y=df['depth'],
            mode='lines',
            name='Salinity (PSU)',
            line=dict(color='blue'),
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=df['current_speed'],
            y=df['depth'],
            mode='lines',
            name='Speed (m/s)',
            line=dict(color='green'),
        ), row=1, col=3)

        fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
        fig.update_xaxes(title_text='Temp (°C)', row=1, col=1)
        fig.update_xaxes(title_text='Salinity', row=1, col=2)
        fig.update_xaxes(title_text='Speed (m/s)', row=1, col=3)

        fig.update_layout(
            title=f'Combined Ocean Profile - {buoy_name}',
            template=self.template,
            height=600,
            showlegend=False,
        )

        return fig

    def create_buoy_location_map(self, buoy_list, surface_data=None):
        fig = go.Figure()

        lats = [b['lat'] for b in buoy_list]
        lons = [b['lon'] for b in buoy_list]
        names = [b['name'] for b in buoy_list]
        ids = [b['id'] for b in buoy_list]

        hover_texts = []
        for i, buoy in enumerate(buoy_list):
            text = f"ID: {buoy['id']}<br>Name: {buoy['name']}"
            if surface_data and buoy['id'] in surface_data:
                sd = surface_data[buoy['id']]
                text += f"<br>Temp: {sd['temperature']:.2f}°C"
                text += f"<br>Salinity: {sd['salinity']:.2f} PSU"
                text += f"<br>Wind: {sd.get('wind_speed', 0):.1f} m/s"
            hover_texts.append(text)

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            text=hover_texts,
            hoverinfo='text',
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                line=dict(width=2, color='white'),
            ),
        ))

        fig.update_layout(
            title='Global Buoy Network',
            geo=dict(
                showland=True,
                showcoastlines=True,
                showcountries=True,
                landcolor='rgb(217, 217, 217)',
                countrycolor='rgb(255, 255, 255)',
                coastlinecolor='rgb(255, 255, 255)',
                projection_type='equirectangular',
            ),
            template=self.template,
            height=700,
        )

        return fig

    def create_time_series(self, data, variable='temperature'):
        if isinstance(data, dict):
            fig = go.Figure()
            for buoy_id, df in data.items():
                if 'timestamp' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[variable],
                        mode='lines',
                        name=f'{buoy_id} - {variable}',
                    ))
        else:
            fig = go.Figure()
            if 'timestamp' in data.columns and 'buoy_id' in data.columns:
                for buoy_id in data['buoy_id'].unique():
                    subset = data[data['buoy_id'] == buoy_id]
                    fig.add_trace(go.Scatter(
                        x=subset['timestamp'],
                        y=subset[variable],
                        mode='lines',
                        name=f'{buoy_id}',
                    ))

        fig.update_layout(
            title=f'{variable.capitalize()} Time Series',
            xaxis_title='Time',
            yaxis_title=variable.capitalize(),
            template=self.template,
            height=500,
        )

        return fig

    def create_quality_control_plot(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature QC', 'Salinity QC'),
            shared_yaxes=True,
        )

        quality_colors = {
            'good': 'green',
            'suspect': 'orange',
            'bad': 'red',
            'missing': 'gray',
        }

        for quality in ['good', 'suspect', 'bad', 'missing']:
            mask = df['quality_flag'] == quality
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, 'temperature'],
                    y=df.loc[mask, 'depth'],
                    mode='markers',
                    name=f'Temp - {quality}',
                    marker=dict(color=quality_colors[quality], size=8),
                ), row=1, col=1)

        for quality in ['good', 'suspect', 'bad', 'missing']:
            mask = df['quality_flag'] == quality
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=df.loc[mask, 'salinity'],
                    y=df.loc[mask, 'depth'],
                    mode='markers',
                    name=f'Sal - {quality}',
                    marker=dict(color=quality_colors[quality], size=8),
                ), row=1, col=2)

        fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
        fig.update_xaxes(title_text='Temperature (°C)', row=1, col=1)
        fig.update_xaxes(title_text='Salinity (PSU)', row=1, col=2)

        fig.update_layout(
            title=f'Quality Control Results - {buoy_name}',
            template=self.template,
            height=600,
        )

        return fig

    def create_data_assimilation_plot(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature Assimilation', 'Salinity Assimilation'),
            shared_yaxes=True,
        )

        fig.add_trace(go.Scatter(
            x=df['temperature'],
            y=df['depth'],
            mode='lines+markers',
            name='Original',
            line=dict(color='blue', width=1),
            marker=dict(size=4),
        ), row=1, col=1)

        if 'temperature_assimilated' in df.columns:
            assimilated_mask = df['data_assimilated'] if 'data_assimilated' in df.columns else pd.Series([False]*len(df))
            fig.add_trace(go.Scatter(
                x=df['temperature_assimilated'],
                y=df['depth'],
                mode='lines',
                name='Assimilated',
                line=dict(color='red', width=2, dash='dash'),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.loc[assimilated_mask, 'temperature_assimilated'],
                y=df.loc[assimilated_mask, 'depth'],
                mode='markers',
                name='Corrected Points',
                marker=dict(color='orange', size=10, symbol='x'),
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['salinity'],
            y=df['depth'],
            mode='lines+markers',
            name='Original',
            line=dict(color='green', width=1),
            marker=dict(size=4),
        ), row=1, col=2)

        if 'salinity_assimilated' in df.columns:
            assimilated_mask = df['data_assimilated'] if 'data_assimilated' in df.columns else pd.Series([False]*len(df))
            fig.add_trace(go.Scatter(
                x=df['salinity_assimilated'],
                y=df['depth'],
                mode='lines',
                name='Assimilated',
                line=dict(color='red', width=2, dash='dash'),
            ), row=1, col=2)
            fig.add_trace(go.Scatter(
                x=df.loc[assimilated_mask, 'salinity_assimilated'],
                y=df.loc[assimilated_mask, 'depth'],
                mode='markers',
                name='Corrected Points',
                marker=dict(color='orange', size=10, symbol='x'),
            ), row=1, col=2)

        fig.update_yaxes(title_text='Depth (m)', autorange='reversed', row=1, col=1)
        fig.update_xaxes(title_text='Temperature (°C)', row=1, col=1)
        fig.update_xaxes(title_text='Salinity (PSU)', row=1, col=2)

        fig.update_layout(
            title=f'Data Assimilation - {buoy_name}',
            template=self.template,
            height=600,
        )

        return fig

    def create_dashboard(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Temperature Profile',
                'Salinity Profile',
                'Current Speed Profile',
                'T-S Diagram',
                'Current Direction',
                'Quality Control',
            ),
            specs=[
                [{}, {}, {}],
                [{}, {}, {}],
            ],
        )

        fig.add_trace(go.Scatter(
            x=df['temperature'], y=df['depth'],
            mode='lines', line=dict(color='red'), name='Temp',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['salinity'], y=df['depth'],
            mode='lines', line=dict(color='blue'), name='Salinity',
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=df['current_speed'], y=df['depth'],
            mode='lines', line=dict(color='green'), name='Speed',
        ), row=1, col=3)

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'
        fig.add_trace(go.Scatter(
            x=df[sal_col], y=df[temp_col],
            mode='markers',
            marker=dict(color=df['depth'], colorscale='Viridis', size=6),
            name='T-S',
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df['current_direction'], y=df['depth'],
            mode='lines', line=dict(color='purple'), name='Direction',
        ), row=2, col=2)

        if 'quality_flag' in df.columns:
            for flag, color in [('good', 'green'), ('suspect', 'orange'), ('bad', 'red')]:
                mask = df['quality_flag'] == flag
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=df.loc[mask, 'temperature'], y=df.loc[mask, 'depth'],
                        mode='markers', marker=dict(color=color, size=6),
                        name=flag,
                    ), row=2, col=3)

        for i in range(1, 4):
            fig.update_yaxes(autorange='reversed', row=1, col=i)
        for i in range(1, 3):
            fig.update_yaxes(autorange='reversed', row=2, col=i)

        fig.update_layout(
            title=f'Ocean Buoy Analysis Dashboard - {buoy_name}',
            template=self.template,
            height=800,
            showlegend=False,
        )

        return fig

    def create_3d_ts_scatter(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

        fig = go.Figure()

        marker_colors = df['depth']

        fig.add_trace(go.Scatter3d(
            x=df[sal_col],
            y=df[temp_col],
            z=df['depth'],
            mode='markers',
            marker=dict(
                size=5,
                color=marker_colors,
                colorscale='Viridis',
                colorbar=dict(title='Depth (m)', x=0.9),
            ),
            text=[f'Depth: {d:.1f}m<br>Temp: {t:.2f}°C<br>Sal: {s:.2f} PSU'
                  for d, t, s in zip(df['depth'], df[temp_col], df[sal_col])],
            hoverinfo='text',
            name='T-S-D Profile',
        ))

        if 'data_assimilated' in df.columns and df['data_assimilated'].any():
            assim_mask = df['data_assimilated']
            fig.add_trace(go.Scatter3d(
                x=df.loc[assim_mask, sal_col],
                y=df.loc[assim_mask, temp_col],
                z=df.loc[assim_mask, 'depth'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond',
                ),
                text='Assimilated Data',
                hoverinfo='text',
                name='Assimilated Points',
            ))

        fig.update_layout(
            title=f'3D T-S-Diagram - {buoy_name}',
            scene=dict(
                xaxis_title='Salinity (PSU)',
                yaxis_title='Temperature (°C)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                ),
            ),
            template=self.template,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig

    def create_3d_current_vectors(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        fig = go.Figure()

        step = max(1, len(df) // 30)
        df_sampled = df.iloc[::step].reset_index(drop=True)

        depths = df_sampled['depth'].values
        u = df_sampled['current_u'].values
        v = df_sampled['current_v'].values
        speed = df_sampled['current_speed'].values

        angles = np.linspace(0, 2 * np.pi, 36)
        for i in range(len(depths)):
            circle_x = u[i] + 0.1 * np.cos(angles)
            circle_y = v[i] + 0.1 * np.sin(angles)
            circle_z = np.full_like(angles, depths[i])

            fig.add_trace(go.Scatter3d(
                x=circle_x,
                y=circle_y,
                z=circle_z,
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.add_trace(go.Cone(
            x=u,
            y=v,
            z=depths,
            u=u * 0.3,
            v=v * 0.3,
            w=np.zeros_like(u),
            colorscale='Jet',
            colorbar=dict(title='Speed (m/s)', x=0.9),
            sizemode='absolute',
            sizeref=0.1,
            anchor='tail',
        ))

        fig.add_trace(go.Scatter3d(
            x=u,
            y=v,
            z=depths,
            mode='markers',
            marker=dict(
                size=4,
                color=speed,
                colorscale='Jet',
            ),
            text=[f'Depth: {d:.1f}m<br>Speed: {s:.3f} m/s<br>Direction: {dir:.1f}°'
                  for d, s, dir in zip(depths, speed, df_sampled['current_direction'])],
            hoverinfo='text',
            showlegend=False,
        ))

        fig.update_layout(
            title=f'3D Current Velocity Vectors - {buoy_name}',
            scene=dict(
                xaxis_title='U Velocity (m/s)',
                yaxis_title='V Velocity (m/s)',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed'),
                camera=dict(
                    eye=dict(x=1.8, y=1.2, z=0.6),
                ),
            ),
            template=self.template,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig

    def create_3d_temperature_volume(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'

        df_sorted = df.sort_values('depth').reset_index(drop=True)
        depths = df_sorted['depth'].values
        temps = df_sorted[temp_col].values

        n_theta = 40
        theta = np.linspace(0, 2 * np.pi, n_theta)
        grid_depth, grid_theta = np.meshgrid(depths, theta)

        temp_grid = np.tile(temps, (n_theta, 1))

        r_base = 1.0
        r = r_base + (temp_grid - temp_grid.min()) / (temp_grid.max() - temp_grid.min() + 1e-6) * 0.5

        x = r * np.cos(grid_theta)
        y = r * np.sin(grid_theta)
        z = grid_depth

        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=temp_grid,
            colorscale='thermal',
            colorbar=dict(title='Temp (°C)', x=0.9),
            name='Temperature',
        ))

        fig.add_trace(go.Scatter3d(
            x=np.zeros_like(temps),
            y=np.zeros_like(temps),
            z=depths,
            mode='markers',
            marker=dict(
                size=3,
                color=temps,
                colorscale='thermal',
            ),
            showlegend=False,
        ))

        fig.update_layout(
            title=f'3D Temperature Volume Profile - {buoy_name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                ),
            ),
            template=self.template,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig

    def create_3d_salinity_volume(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

        df_sorted = df.sort_values('depth').reset_index(drop=True)
        depths = df_sorted['depth'].values
        sals = df_sorted[sal_col].values

        n_theta = 40
        theta = np.linspace(0, 2 * np.pi, n_theta)
        grid_depth, grid_theta = np.meshgrid(depths, theta)

        sal_grid = np.tile(sals, (n_theta, 1))

        r_base = 1.0
        r = r_base + (sal_grid - sal_grid.min()) / (sal_grid.max() - sal_grid.min() + 1e-6) * 0.5

        x = r * np.cos(grid_theta)
        y = r * np.sin(grid_theta)
        z = grid_depth

        fig = go.Figure()

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=sal_grid,
            colorscale='Viridis',
            colorbar=dict(title='Salinity (PSU)', x=0.9),
            name='Salinity',
        ))

        fig.update_layout(
            title=f'3D Salinity Volume Profile - {buoy_name}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Depth (m)',
                zaxis=dict(autorange='reversed'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),
                ),
            ),
            template=self.template,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig

    def create_3d_multi_buoy_map(self, buoy_list, surface_data=None):
        fig = go.Figure()

        lats = [b['lat'] for b in buoy_list]
        lons = [b['lon'] for b in buoy_list]
        names = [b['name'] for b in buoy_list]
        ids = [b['id'] for b in buoy_list]

        temps = []
        for buoy in buoy_list:
            if surface_data and buoy['id'] in surface_data:
                temps.append(surface_data[buoy['id']].get('temperature', 15))
            else:
                temps.append(15)

        heights = [0.5 + (t - min(temps)) / (max(temps) - min(temps) + 1e-6) * 2 for t in temps]

        hover_texts = []
        for i, buoy in enumerate(buoy_list):
            text = f"ID: {buoy['id']}<br>Name: {buoy['name']}<br>Lat: {buoy['lat']}<br>Lon: {buoy['lon']}"
            if surface_data and buoy['id'] in surface_data:
                sd = surface_data[buoy['id']]
                text += f"<br>Temp: {sd.get('temperature', 'N/A'):.2f}°C"
                text += f"<br>Salinity: {sd.get('salinity', 'N/A'):.2f} PSU"
            hover_texts.append(text)

        for i in range(len(buoy_list)):
            fig.add_trace(go.Scatter3d(
                x=[lons[i], lons[i]],
                y=[lats[i], lats[i]],
                z=[0, heights[i]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.add_trace(go.Scatter3d(
            x=lons,
            y=lats,
            z=heights,
            mode='markers',
            marker=dict(
                size=8,
                color=temps,
                colorscale='thermal',
                colorbar=dict(title='SST (°C)', x=0.9),
                line=dict(width=2, color='white'),
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Buoys',
        ))

        fig.update_layout(
            title='3D Global Buoy Network Map',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='SST Anomaly',
                xaxis=dict(range=[-180, 180]),
                yaxis=dict(range=[-90, 90]),
                zaxis=dict(range=[0, 3]),
                camera=dict(
                    eye=dict(x=0.8, y=1.2, z=0.6),
                ),
            ),
            template=self.template,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        return fig

    def create_3d_dashboard(self, data, buoy_id=None):
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

        fig = go.Figure()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '3D T-S Profile',
                '3D Temperature Volume',
                '3D Current Vectors',
                '3D Salinity Volume',
            ),
            specs=[
                [{'type': 'scene'}, {'type': 'scene'}],
                [{'type': 'scene'}, {'type': 'scene'}],
            ],
        )

        fig.add_trace(go.Scatter3d(
            x=df[sal_col], y=df[temp_col], z=df['depth'],
            mode='markers',
            marker=dict(size=3, color=df['depth'], colorscale='Viridis'),
        ), row=1, col=1)

        df_sorted = df.sort_values('depth').reset_index(drop=True)
        depths = df_sorted['depth'].values
        temps = df_sorted[temp_col].values
        n_theta = 20
        theta = np.linspace(0, 2 * np.pi, n_theta)
        gd, gt = np.meshgrid(depths, theta)
        tg = np.tile(temps, (n_theta, 1))
        r = 1.0 + (tg - tg.min()) / (tg.max() - tg.min() + 1e-6) * 0.5
        x = r * np.cos(gt)
        y = r * np.sin(gt)

        fig.add_trace(go.Surface(
            x=x, y=y, z=gd,
            surfacecolor=tg,
            colorscale='thermal',
        ), row=1, col=2)

        step = max(1, len(df) // 20)
        df_s = df.iloc[::step].reset_index(drop=True)
        fig.add_trace(go.Cone(
            x=df_s['current_u'], y=df_s['current_v'], z=df_s['depth'],
            u=df_s['current_u'] * 0.3, v=df_s['current_v'] * 0.3,
            w=np.zeros(len(df_s)),
            colorscale='Jet',
            sizemode='absolute',
            sizeref=0.08,
            anchor='tail',
        ), row=2, col=1)

        sals = df_sorted[sal_col].values
        sg = np.tile(sals, (n_theta, 1))
        r2 = 1.0 + (sg - sg.min()) / (sg.max() - sg.min() + 1e-6) * 0.5
        x2 = r2 * np.cos(gt)
        y2 = r2 * np.sin(gt)

        fig.add_trace(go.Surface(
            x=x2, y=y2, z=gd,
            surfacecolor=sg,
            colorscale='Viridis',
        ), row=2, col=2)

        for r_idx in range(1, 3):
            for c_idx in range(1, 3):
                fig.update_scenes(
                    dict(zaxis=dict(autorange='reversed')),
                    row=r_idx, col=c_idx
                )

        fig.update_layout(
            title=f'3D Ocean Profile Dashboard - {buoy_name}',
            template=self.template,
            height=800,
            showlegend=False,
        )

        return fig

    def _extract_single_buoy(self, data, buoy_id):
        if isinstance(data, dict):
            if buoy_id:
                return data[buoy_id]
            else:
                first_key = list(data.keys())[0]
                return data[first_key]
        elif isinstance(data, pd.DataFrame):
            if buoy_id and 'buoy_id' in data.columns:
                return data[data['buoy_id'] == buoy_id].copy()
            else:
                return data.copy()
        return data
