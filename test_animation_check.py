from main import OceanBuoySystem
from visualization import OceanVisualizer

system = OceanBuoySystem(use_simulation=True, assimilation_method='optimal')
system.run_offline_analysis()

time_series = system._generate_demo_time_series(num_steps=5)
print(f'Time series generated: {len(time_series)} steps')
print(f'First step columns: {list(time_series[0].columns)}')
print('Demo time series generation: OK')

viz = OceanVisualizer()
print(f'Has generate_profile_animation: {hasattr(viz, "generate_profile_animation")}')
print(f'Has generate_ts_diagram_animation: {hasattr(viz, "generate_ts_diagram_animation")}')
print(f'Has generate_multi_panel_animation: {hasattr(viz, "generate_multi_panel_animation")}')
print(f'Has generate_buoy_map_animation: {hasattr(viz, "generate_buoy_map_animation")}')
print('All animation methods exist: OK')

import matplotlib
print(f'Matplotlib available: {matplotlib.__version__}')
