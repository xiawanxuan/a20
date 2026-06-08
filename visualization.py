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

    def create_thermocline_plot(self, data, analysis_results=None, buoy_id=None):
        from ocean_analysis import OceanAnalyzer
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        analyzer = OceanAnalyzer()
        if analysis_results is None:
            analysis_results = analyzer.analyze_profile(df)

        thermocline = analysis_results.get('thermocline', {})

        fig = go.Figure()

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        fig.add_trace(go.Scatter(
            x=df[temp_col],
            y=df['depth'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='red', width=2),
            marker=dict(size=4),
        ))

        if thermocline.get('found', False):
            fig.add_vline(
                x=df[temp_col].iloc[0] if thermocline['depth'] > df['depth'].max() else df[temp_col].iloc[0],
                line_dash="dash",
            )
            fig.add_hline(
                y=thermocline['depth'],
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Thermocline: {thermocline['depth']:.0f}m ({thermocline['strength'].replace('_', ' ')})",
                annotation_position="bottom right",
            )
            fig.add_hrect(
                y0=thermocline['top_depth'],
                y1=thermocline['bottom_depth'],
                fillcolor="orange",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        fig.update_layout(
            title=f'Temperature Profile with Thermocline - {buoy_name}',
            xaxis_title='Temperature (°C)',
            yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            template=self.template,
            height=600,
        )

        return fig

    def create_halocline_plot(self, data, analysis_results=None, buoy_id=None):
        from ocean_analysis import OceanAnalyzer
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        analyzer = OceanAnalyzer()
        if analysis_results is None:
            analysis_results = analyzer.analyze_profile(df)

        halocline = analysis_results.get('halocline', {})

        fig = go.Figure()

        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'
        fig.add_trace(go.Scatter(
            x=df[sal_col],
            y=df['depth'],
            mode='lines+markers',
            name='Salinity',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
        ))

        if halocline.get('found', False):
            fig.add_hline(
                y=halocline['depth'],
                line_dash="dash",
                line_color="cyan",
                annotation_text=f"Halocline: {halocline['depth']:.0f}m ({halocline['strength'].replace('_', ' ')})",
                annotation_position="bottom right",
            )
            fig.add_hrect(
                y0=halocline['top_depth'],
                y1=halocline['bottom_depth'],
                fillcolor="cyan",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        fig.update_layout(
            title=f'Salinity Profile with Halocline - {buoy_name}',
            xaxis_title='Salinity (PSU)',
            yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            template=self.template,
            height=600,
        )

        return fig

    def create_stratification_plot(self, data, analysis_results=None, buoy_id=None):
        from ocean_analysis import OceanAnalyzer
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        analyzer = OceanAnalyzer()
        if analysis_results is None:
            analysis_results = analyzer.analyze_profile(df)

        pycnocline = analysis_results.get('pycnocline', {})
        mld = analysis_results.get('mixed_layer_depth', {})

        fig = go.Figure()

        if pycnocline.get('found', False) and 'buoyancy_frequency' in pycnocline:
            depths = df['depth'].values
            n2 = np.array(pycnocline['buoyancy_frequency'])

            fig.add_trace(go.Scatter(
                x=n2,
                y=depths,
                mode='lines',
                name='Buoyancy Frequency (N)',
                line=dict(color='purple', width=2),
                fill='tozerox',
            ))

            if pycnocline.get('found', False):
                fig.add_hline(
                    y=pycnocline['depth'],
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Pycnocline: {pycnocline['depth']:.0f}m",
                    annotation_position="bottom right",
                )

        if mld.get('found', False):
            fig.add_hline(
                y=mld['depth'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"MLD: {mld['depth']:.0f}m",
                annotation_position="top right",
            )

        fig.update_layout(
            title=f'Water Column Stratification - {buoy_name}',
            xaxis_title='Buoyancy Frequency (rad/s)',
            yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            template=self.template,
            height=600,
        )

        return fig

    def create_eddy_detection_plot(self, data, eddies=None, buoy_id=None):
        from ocean_analysis import OceanAnalyzer
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        analyzer = OceanAnalyzer()
        if eddies is None:
            eddies = analyzer.detect_eddies(df)

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=('Temperature Anomalies', 'Salinity Anomalies'),
            shared_yaxes=True,
        )

        temps = df[temp_col].values
        sals = df[sal_col].values
        depths = df['depth'].values

        from scipy.ndimage import gaussian_filter1d
        temp_smooth = gaussian_filter1d(temps, sigma=2.0)
        sal_smooth = gaussian_filter1d(sals, sigma=2.0)
        temp_anomaly = temps - temp_smooth
        sal_anomaly = sals - sal_smooth

        fig.add_trace(go.Scatter(
            x=temp_anomaly,
            y=depths,
            mode='lines+markers',
            name='Temp Anomaly',
            line=dict(color='red', width=2),
            marker=dict(size=4),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=sal_anomaly,
            y=depths,
            mode='lines+markers',
            name='Sal Anomaly',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
        ), row=1, col=2)

        for i, eddy in enumerate(eddies):
            color = 'red' if 'warm' in eddy['type'] else 'blue'
            fig.add_hrect(
                y0=eddy['top_depth'],
                y1=eddy['bottom_depth'],
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=f"Eddy {i+1}: {eddy['type'].replace('_', ' ')}",
                annotation_position="left",
                row=1, col=1,
            )
            fig.add_hrect(
                y0=eddy['top_depth'],
                y1=eddy['bottom_depth'],
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=2,
            )

        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)

        fig.update_layout(
            title=f'Eddy Detection - {buoy_name} ({len(eddies)} eddies found)',
            yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            template=self.template,
            height=600,
            showlegend=False,
        )
        fig.update_xaxes(title_text='Temperature Anomaly (°C)', row=1, col=1)
        fig.update_xaxes(title_text='Salinity Anomaly (PSU)', row=1, col=2)

        return fig

    def create_water_masses_plot(self, data, analysis_results=None, buoy_id=None):
        from ocean_analysis import OceanAnalyzer
        df = self._extract_single_buoy(data, buoy_id)
        buoy_name = df['buoy_name'].iloc[0] if 'buoy_name' in df.columns else buoy_id

        analyzer = OceanAnalyzer()
        if analysis_results is None:
            analysis_results = analyzer.analyze_profile(df)

        water_masses = analysis_results.get('water_masses', [])

        fig = go.Figure()

        temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
        sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        for i, mass in enumerate(water_masses):
            mask = (df['depth'] >= mass['top_depth']) & (df['depth'] <= mass['bottom_depth'])
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=df.loc[mask, sal_col],
                y=df.loc[mask, temp_col],
                mode='markers',
                name=mass['name'],
                marker=dict(
                    size=6,
                    color=color,
                ),
                text=[f"Depth: {d:.0f}m<br>Temp: {t:.2f}°C<br>Sal: {s:.2f} PSU"
                      for d, t, s in zip(df.loc[mask, 'depth'], df.loc[mask, temp_col], df.loc[mask, sal_col])],
                hoverinfo='text',
            ))

        fig.update_layout(
            title=f'Water Mass Classification - {buoy_name}',
            xaxis_title='Salinity (PSU)',
            yaxis_title='Temperature (°C)',
            template=self.template,
            height=600,
            legend_title='Water Masses',
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

    def generate_profile_animation(self, time_series_data, variable='temperature',
                                    output_path='profile_animation.mp4',
                                    fps=10, dpi=100, figsize=(10, 8)):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib import cm
        except ImportError:
            print('[WARNING] matplotlib not available, cannot generate MP4 animation')
            return None

        if not time_series_data or len(time_series_data) < 2:
            print('[WARNING] Insufficient time series data for animation')
            return None

        first_df = self._extract_single_buoy(time_series_data[0], None)
        buoy_name = first_df['buoy_name'].iloc[0] if 'buoy_name' in first_df.columns else 'Buoy'

        all_depths = first_df['depth'].values
        max_depth = all_depths.max()

        all_values = []
        all_times = []
        for i, ts_data in enumerate(time_series_data):
            df = self._extract_single_buoy(ts_data, None)
            if 'timestamp' in df.columns:
                all_times.append(str(df['timestamp'].iloc[0]))
            else:
                all_times.append(f'Time Step {i + 1}')

            if variable + '_assimilated' in df.columns:
                all_values.append(df[variable + '_assimilated'].values)
            else:
                all_values.append(df[variable].values)

        all_values = np.array(all_values)

        fig, ax = plt.subplots(figsize=figsize)
        line, = ax.plot([], [], 'b-', linewidth=2, label=variable.capitalize())
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if variable == 'temperature':
            color = 'red'
            xlabel = 'Temperature (°C)'
            cmap = cm.thermal
        elif variable == 'salinity':
            color = 'blue'
            xlabel = 'Salinity (PSU)'
            cmap = cm.viridis
        else:
            color = 'green'
            xlabel = variable.capitalize()
            cmap = cm.cividis

        line.set_color(color)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_title(f'{variable.capitalize()} Profile Evolution - {buoy_name}', fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        vmin = np.nanmin(all_values)
        vmax = np.nanmax(all_values)
        x_margin = (vmax - vmin) * 0.1 if vmax != vmin else 1.0
        ax.set_xlim(vmin - x_margin, vmax + x_margin)
        ax.set_ylim(max_depth, 0)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def update(frame):
            line.set_data(all_values[frame], all_depths)
            time_text.set_text(f'Time: {all_times[frame]}')
            return line, time_text

        anim = FuncAnimation(
            fig, update, frames=len(time_series_data),
            init_func=init, blit=True, interval=1000 / fps
        )

        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=2000)
            print(f'[OK] Animation saved to: {output_path}')
        except Exception as e:
            try:
                anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
                print(f'[OK] Animation saved (GIF): {output_path}')
            except Exception as e2:
                print(f'[ERROR] Failed to save animation: {e}, {e2}')
                plt.close(fig)
                return None

        plt.close(fig)
        return output_path

    def generate_ts_diagram_animation(self, time_series_data,
                                       output_path='ts_animation.mp4',
                                       fps=10, dpi=100, figsize=(10, 8)):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib import cm
        except ImportError:
            print('[WARNING] matplotlib not available, cannot generate MP4 animation')
            return None

        if not time_series_data or len(time_series_data) < 2:
            print('[WARNING] Insufficient time series data for animation')
            return None

        first_df = self._extract_single_buoy(time_series_data[0], None)
        buoy_name = first_df['buoy_name'].iloc[0] if 'buoy_name' in first_df.columns else 'Buoy'

        all_temps = []
        all_sals = []
        all_depths = []
        all_times = []

        for i, ts_data in enumerate(time_series_data):
            df = self._extract_single_buoy(ts_data, None)
            if 'timestamp' in df.columns:
                all_times.append(str(df['timestamp'].iloc[0]))
            else:
                all_times.append(f'Time Step {i + 1}')

            temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
            sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

            all_temps.append(df[temp_col].values)
            all_sals.append(df[sal_col].values)
            all_depths.append(df['depth'].values)

        all_temps = np.array(all_temps)
        all_sals = np.array(all_sals)
        depths_ref = all_depths[0]

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter([], [], c=[], cmap=cm.viridis, s=30, alpha=0.7)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Depth (m)')

        ax.set_xlabel('Salinity (PSU)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f'T-S Diagram Evolution - {buoy_name}', fontsize=14)
        ax.grid(True, alpha=0.3)

        x_min, x_max = np.nanmin(all_sals), np.nanmax(all_sals)
        y_min, y_max = np.nanmin(all_temps), np.nanmax(all_temps)
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        def init():
            scatter.set_offsets(np.column_stack(([], [])))
            scatter.set_array(np.array([]))
            time_text.set_text('')
            return scatter, time_text

        def update(frame):
            pts = np.column_stack((all_sals[frame], all_temps[frame]))
            scatter.set_offsets(pts)
            scatter.set_array(depths_ref)
            time_text.set_text(f'Time: {all_times[frame]}')
            return scatter, time_text

        anim = FuncAnimation(
            fig, update, frames=len(time_series_data),
            init_func=init, blit=False, interval=1000 / fps
        )

        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=2000)
            print(f'[OK] T-S animation saved to: {output_path}')
        except Exception as e:
            try:
                anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
                print(f'[OK] T-S animation saved (GIF): {output_path}')
            except Exception as e2:
                print(f'[ERROR] Failed to save T-S animation: {e}, {e2}')
                plt.close(fig)
                return None

        plt.close(fig)
        return output_path

    def generate_multi_panel_animation(self, time_series_data,
                                       output_path='multi_panel_animation.mp4',
                                       fps=10, dpi=100, figsize=(14, 10)):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib import cm
        except ImportError:
            print('[WARNING] matplotlib not available, cannot generate MP4 animation')
            return None

        if not time_series_data or len(time_series_data) < 2:
            print('[WARNING] Insufficient time series data for animation')
            return None

        first_df = self._extract_single_buoy(time_series_data[0], None)
        buoy_name = first_df['buoy_name'].iloc[0] if 'buoy_name' in first_df.columns else 'Buoy'

        all_temps = []
        all_sals = []
        all_speeds = []
        all_depths = []
        all_times = []

        for i, ts_data in enumerate(time_series_data):
            df = self._extract_single_buoy(ts_data, None)
            if 'timestamp' in df.columns:
                all_times.append(str(df['timestamp'].iloc[0]))
            else:
                all_times.append(f'Time Step {i + 1}')

            temp_col = 'temperature_assimilated' if 'temperature_assimilated' in df.columns else 'temperature'
            sal_col = 'salinity_assimilated' if 'salinity_assimilated' in df.columns else 'salinity'

            all_temps.append(df[temp_col].values)
            all_sals.append(df[sal_col].values)
            all_speeds.append(df['current_speed'].values if 'current_speed' in df.columns else np.zeros(len(df)))
            all_depths.append(df['depth'].values)

        all_temps = np.array(all_temps)
        all_sals = np.array(all_sals)
        all_speeds = np.array(all_speeds)
        depths_ref = all_depths[0]
        max_depth = depths_ref.max()

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        line1, = ax1.plot([], [], 'r-', linewidth=2)
        line2, = ax2.plot([], [], 'b-', linewidth=2)
        line3, = ax3.plot([], [], 'g-', linewidth=2)
        scatter4 = ax4.scatter([], [], c=[], cmap=cm.viridis, s=20, alpha=0.7)

        time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title('Temperature Profile')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Salinity (PSU)')
        ax2.set_ylabel('Depth (m)')
        ax2.set_title('Salinity Profile')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        ax3.set_xlabel('Current Speed (m/s)')
        ax3.set_ylabel('Depth (m)')
        ax3.set_title('Current Speed Profile')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)

        ax4.set_xlabel('Salinity (PSU)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('T-S Diagram')
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter4, ax=ax4)
        cbar.set_label('Depth (m)')

        t_min, t_max = np.nanmin(all_temps), np.nanmax(all_temps)
        s_min, s_max = np.nanmin(all_sals), np.nanmax(all_sals)
        sp_min, sp_max = np.nanmin(all_speeds), np.nanmax(all_speeds)

        ax1.set_xlim(t_min - 0.5, t_max + 0.5)
        ax1.set_ylim(max_depth, 0)
        ax2.set_xlim(s_min - 0.2, s_max + 0.2)
        ax2.set_ylim(max_depth, 0)
        ax3.set_xlim(max(0, sp_min - 0.1), sp_max + 0.1)
        ax3.set_ylim(max_depth, 0)
        ax4.set_xlim(s_min - 0.2, s_max + 0.2)
        ax4.set_ylim(t_min - 0.5, t_max + 0.5)

        fig.suptitle(f'Ocean Data Evolution - {buoy_name}', fontsize=16, y=0.98)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            scatter4.set_offsets(np.column_stack(([], [])))
            scatter4.set_array(np.array([]))
            time_text.set_text('')
            return line1, line2, line3, scatter4, time_text

        def update(frame):
            line1.set_data(all_temps[frame], depths_ref)
            line2.set_data(all_sals[frame], depths_ref)
            line3.set_data(all_speeds[frame], depths_ref)
            pts = np.column_stack((all_sals[frame], all_temps[frame]))
            scatter4.set_offsets(pts)
            scatter4.set_array(depths_ref)
            time_text.set_text(f'Time: {all_times[frame]}')
            return line1, line2, line3, scatter4, time_text

        anim = FuncAnimation(
            fig, update, frames=len(time_series_data),
            init_func=init, blit=False, interval=1000 / fps
        )

        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=2500)
            print(f'[OK] Multi-panel animation saved to: {output_path}')
        except Exception as e:
            try:
                anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
                print(f'[OK] Multi-panel animation saved (GIF): {output_path}')
            except Exception as e2:
                print(f'[ERROR] Failed to save multi-panel animation: {e}, {e2}')
                plt.close(fig)
                return None

        plt.close(fig)
        return output_path

    def generate_buoy_map_animation(self, time_series_surface_data, buoy_list,
                                     output_path='buoy_map_animation.mp4',
                                     fps=5, dpi=100, figsize=(12, 8)):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            from matplotlib import cm
        except ImportError:
            print('[WARNING] matplotlib not available, cannot generate MP4 animation')
            return None

        if not time_series_surface_data or len(time_series_surface_data) < 2:
            print('[WARNING] Insufficient time series data for animation')
            return None

        all_times = []
        all_temps = []
        all_sals = []

        for i, data in enumerate(time_series_surface_data):
            all_times.append(f'Time Step {i + 1}')
            if isinstance(data, list):
                all_temps.append([d.get('surface_temperature', 0) for d in data])
                all_sals.append([d.get('surface_salinity', 0) for d in data])
            else:
                all_temps.append(data if isinstance(data, list) else [data])
                all_sals.append(data if isinstance(data, list) else [data])

        lats = [b['lat'] for b in buoy_list]
        lons = [b['lon'] for b in buoy_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        scatter1 = ax1.scatter(lons, lats, c=all_temps[0], cmap=cm.thermal, s=100, edgecolors='black')
        scatter2 = ax2.scatter(lons, lats, c=all_sals[0], cmap=cm.viridis, s=100, edgecolors='black')

        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('SST (°C)')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('SSS (PSU)')

        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Sea Surface Temperature')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Sea Surface Salinity')
        ax2.grid(True, alpha=0.3)

        time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        t_min, t_max = np.min(all_temps), np.max(all_temps)
        s_min, s_max = np.min(all_sals), np.max(all_sals)
        scatter1.set_clim(t_min, t_max)
        scatter2.set_clim(s_min, s_max)

        fig.suptitle('Global Buoy Surface Data Evolution', fontsize=16, y=0.98)

        def update(frame):
            scatter1.set_array(np.array(all_temps[frame]))
            scatter2.set_array(np.array(all_sals[frame]))
            time_text.set_text(f'Time: {all_times[frame]}')
            return scatter1, scatter2, time_text

        anim = FuncAnimation(
            fig, update, frames=len(time_series_surface_data),
            blit=False, interval=1000 / fps
        )

        try:
            anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi, bitrate=2000)
            print(f'[OK] Buoy map animation saved to: {output_path}')
        except Exception as e:
            try:
                anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
                print(f'[OK] Buoy map animation saved (GIF): {output_path}')
            except Exception as e2:
                print(f'[ERROR] Failed to save buoy map animation: {e}, {e2}')
                plt.close(fig)
                return None

        plt.close(fig)
        return output_path
