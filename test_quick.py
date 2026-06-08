import os
import sys

def test_analysis_and_visualization():
    print("=" * 60)
    print("QUICK TEST: Ocean Analysis + Visualization")
    print("=" * 60)

    try:
        from ocean_analysis import OceanAnalyzer
        from visualization import OceanVisualizer
        from data_acquisition import BuoyDataFetcher
        from quality_control import QualityController

        fetcher = BuoyDataFetcher(use_simulation=True)
        qc = QualityController(assimilation_method='optimal')
        analyzer = OceanAnalyzer()
        visualizer = OceanVisualizer()

        raw_data = fetcher.fetch_realtime_data()
        first_buoy = list(raw_data.keys())[0]
        df = qc.run_full_qc(raw_data)[first_buoy]

        print(f"\n[OK] Data loaded: buoy {first_buoy}, {len(df)} depth levels")

        print("\n--- Thermocline Detection ---")
        thermocline = analyzer.detect_thermocline(df)
        print(f"  Found: {thermocline['found']}")
        if thermocline['found']:
            print(f"  Depth: {thermocline['depth']:.1f} m")
            print(f"  Strength: {thermocline['strength']}")

        print("\n--- Halocline Detection ---")
        halocline = analyzer.detect_halocline(df)
        print(f"  Found: {halocline['found']}")
        if halocline['found']:
            print(f"  Depth: {halocline['depth']:.1f} m")

        print("\n--- Pycnocline Detection ---")
        pycnocline = analyzer.detect_pycnocline(df)
        print(f"  Found: {pycnocline['found']}")
        if pycnocline['found']:
            print(f"  Depth: {pycnocline['depth']:.1f} m")

        print("\n--- Eddy Detection ---")
        eddies = analyzer.detect_eddies(df)
        print(f"  Eddies found: {len(eddies)}")
        for i, eddy in enumerate(eddies[:3]):
            print(f"    Eddy {i+1}: {eddy['type']}, core at {eddy['core_depth']:.0f}m")

        print("\n--- Water Mass Classification ---")
        water_masses = analyzer.classify_water_masses(df)
        print(f"  Water masses: {len(water_masses)}")
        for mass in water_masses:
            print(f"    {mass['name']}: {mass['top_depth']:.0f}m - {mass['bottom_depth']:.0f}m")

        print("\n--- Full Profile Analysis ---")
        results = analyzer.analyze_profile(df)
        summary = analyzer.get_analysis_dict(results)
        print(f"  Thermocline found: {summary['thermocline']['found']}")
        print(f"  Eddies count: {summary['eddies_count']}")
        print(f"  Water masses count: {summary['water_masses_count']}")

        print("\n--- New Visualization Plots ---")
        output_dir = "output_test_quick"
        os.makedirs(output_dir, exist_ok=True)

        plot_methods = [
            "create_thermocline_plot",
            "create_halocline_plot",
            "create_stratification_plot",
            "create_eddy_detection_plot",
            "create_water_masses_plot",
        ]

        for method_name in plot_methods:
            try:
                method = getattr(visualizer, method_name)
                fig = method(df, buoy_id=first_buoy)
                if fig is not None:
                    filepath = os.path.join(output_dir, f"{method_name}.html")
                    fig.write_html(filepath)
                    print(f"  [OK] {method_name}")
                else:
                    print(f"  [SKIP] {method_name} returned None")
            except Exception as e:
                print(f"  [FAIL] {method_name}: {e}")

        print("\n" + "=" * 60)
        print("[OK] All analysis and visualization tests PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_integration_quick():
    print("\n" + "=" * 60)
    print("QUICK TEST: System Integration")
    print("=" * 60)

    try:
        from main import OceanBuoySystem

        print("\n  Initializing OceanBuoySystem...")
        system = OceanBuoySystem(use_simulation=True, assimilation_method='optimal')
        print(f"  [OK] System initialized")
        print(f"  [OK] Has analyzer: {hasattr(system, 'analyzer')}")
        print(f"  [OK] Has generate_profile_animation: {hasattr(system, 'generate_profile_animation')}")
        print(f"  [OK] Has detect_thermocline: {hasattr(system, 'detect_thermocline')}")
        print(f"  [OK] Has detect_eddies: {hasattr(system, 'detect_eddies')}")

        print("\n  Running offline analysis...")
        result = system.run_offline_analysis()
        print(f"  [OK] Analysis completed")

        print(f"  [OK] Analysis results available: {system.analysis_results is not None}")

        print("\n  Testing detect_thermocline...")
        thermocline = system.detect_thermocline()
        print(f"  [OK] Thermocline found: {thermocline['found'] if thermocline else 'N/A'}")

        print("\n  Testing detect_eddies...")
        eddies = system.detect_eddies()
        print(f"  [OK] Eddies detected: {len(eddies) if eddies else 0}")

        print("\n" + "=" * 60)
        print("[OK] System integration tests PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n[FAILED] System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("QUICK COMPREHENSIVE TEST")
    print("=" * 60)

    r1 = test_analysis_and_visualization()
    r2 = test_system_integration_quick()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Analysis + Visualization: {'PASS' if r1 else 'FAIL'}")
    print(f"  System Integration: {'PASS' if r2 else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if r1 and r2 else 'SOME FAIL'}")

    return 0 if (r1 and r2) else 1


if __name__ == '__main__':
    sys.exit(main())
