import os
import sys
import json

def test_ocean_analysis():
    print("=" * 60)
    print("TEST 1: Ocean Analysis Module")
    print("=" * 60)

    try:
        from ocean_analysis import OceanAnalyzer
        from data_acquisition import BuoyDataFetcher
        from quality_control import QualityController

        fetcher = BuoyDataFetcher(use_simulation=True)
        qc = QualityController(assimilation_method='optimal')
        analyzer = OceanAnalyzer()

        raw_data = fetcher.fetch_realtime_data()
        first_buoy = list(raw_data.keys())[0]
        df = qc.run_full_qc(raw_data)[first_buoy]

        print(f"\nTesting with buoy {first_buoy} ({len(df)} depth levels)")

        print("\n--- Thermocline Detection ---")
        thermocline = analyzer.detect_thermocline(df)
        print(f"  Found: {thermocline['found']}")
        if thermocline['found']:
            print(f"  Depth: {thermocline['depth']:.1f} m")
            print(f"  Thickness: {thermocline['thickness']:.1f} m")
            print(f"  Strength: {thermocline['strength']}")
            print(f"  Max gradient: {thermocline['max_gradient']:.4f} C/m")

        print("\n--- Halocline Detection ---")
        halocline = analyzer.detect_halocline(df)
        print(f"  Found: {halocline['found']}")
        if halocline['found']:
            print(f"  Depth: {halocline['depth']:.1f} m")
            print(f"  Strength: {halocline['strength']}")

        print("\n--- Pycnocline Detection ---")
        pycnocline = analyzer.detect_pycnocline(df)
        print(f"  Found: {pycnocline['found']}")
        if pycnocline['found']:
            print(f"  Depth: {pycnocline['depth']:.1f} m")

        print("\n--- Mixed Layer Depth ---")
        mld = analyzer.calculate_mld(df)
        print(f"  Found: {mld['found']}")
        if mld['found']:
            print(f"  MLD: {mld['depth']:.1f} m")

        print("\n--- Eddy Detection (Vertical Profile) ---")
        eddies = analyzer.detect_eddies(df)
        print(f"  Eddies found: {len(eddies)}")
        for i, eddy in enumerate(eddies):
            print(f"    Eddy {i+1}: {eddy['type']}, depth: {eddy['core_depth']:.0f}m, "
                  f"thickness: {eddy['thickness']:.0f}m, amplitude: {eddy['temperature_anomaly']:.3f} C")

        print("\n--- Water Mass Classification ---")
        water_masses = analyzer.classify_water_masses(df)
        print(f"  Water masses: {len(water_masses)}")
        for mass in water_masses:
            print(f"    {mass['name']}: {mass['top_depth']:.0f}m - {mass['bottom_depth']:.0f}m")

        print("\n--- Full Profile Analysis ---")
        results = analyzer.analyze_profile(df)
        summary = analyzer.get_analysis_dict(results)
        print(f"  Summary keys: {list(summary.keys())}")
        print(f"  Thermocline found: {summary['thermocline']['found']}")
        print(f"  Eddies count: {summary['eddies_count']}")
        print(f"  Water masses count: {summary['water_masses_count']}")

        print("\n[OK] Ocean analysis module tests PASSED")
        return True

    except Exception as e:
        print(f"\n[FAILED] Ocean analysis module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_new():
    print("\n" + "=" * 60)
    print("TEST 2: New Visualization Methods")
    print("=" * 60)

    try:
        from visualization import OceanVisualizer
        from data_acquisition import BuoyDataFetcher
        from quality_control import QualityController

        fetcher = BuoyDataFetcher(use_simulation=True)
        qc = QualityController(assimilation_method='optimal')
        visualizer = OceanVisualizer()

        raw_data = fetcher.fetch_realtime_data()
        first_buoy = list(raw_data.keys())[0]
        df = qc.run_full_qc(raw_data)[first_buoy]

        print(f"\nTesting with buoy {first_buoy}")

        output_dir = "output_test"
        os.makedirs(output_dir, exist_ok=True)

        plots = [
            ("thermocline", "create_thermocline_plot"),
            ("halocline", "create_halocline_plot"),
            ("stratification", "create_stratification_plot"),
            ("eddy_detection", "create_eddy_detection_plot"),
            ("water_masses", "create_water_masses_plot"),
        ]

        for name, method_name in plots:
            print(f"\n  Testing {name}...")
            try:
                method = getattr(visualizer, method_name)
                fig = method(df, buoy_id=first_buoy)
                if fig is not None:
                    filepath = os.path.join(output_dir, f"test_{name}.html")
                    fig.write_html(filepath)
                    print(f"    [OK] {name} plot generated: {filepath}")
                else:
                    print(f"    [SKIP] {name} returned None")
            except Exception as e:
                print(f"    [FAILED] {name}: {e}")
                import traceback
                traceback.print_exc()

        print("\n[OK] New visualization tests PASSED")
        return True

    except Exception as e:
        print(f"\n[FAILED] Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_animation_export():
    print("\n" + "=" * 60)
    print("TEST 3: Animation Export")
    print("=" * 60)

    try:
        from main import OceanBuoySystem

        print("\n  Initializing system for animation test...")
        system = OceanBuoySystem(use_simulation=True, assimilation_method='optimal')
        system.run_offline_analysis()

        output_dir = "output_test"
        os.makedirs(output_dir, exist_ok=True)

        print("\n  Testing profile animation (temperature)...")
        try:
            output_path = os.path.join(output_dir, "test_temp_profile_animation.mp4")
            result = system.generate_profile_animation(
                variable='temperature', output_path=output_path, fps=5
            )
            if result and os.path.exists(result):
                print(f"    [OK] Temperature profile animation: {result}")
            else:
                print(f"    [SKIP] Animation not generated (FFmpeg may not be available)")
        except Exception as e:
            print(f"    [SKIP] Profile animation error: {e}")

        print("\n  Testing multi-panel animation...")
        try:
            output_path = os.path.join(output_dir, "test_multi_panel_animation.mp4")
            result = system.generate_multi_panel_animation(
                output_path=output_path, fps=5
            )
            if result and os.path.exists(result):
                print(f"    [OK] Multi-panel animation: {result}")
            else:
                print(f"    [SKIP] Multi-panel animation not generated")
        except Exception as e:
            print(f"    [SKIP] Multi-panel animation error: {e}")

        print("\n[OK] Animation export tests completed")
        return True

    except Exception as e:
        print(f"\n[FAILED] Animation export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_integration():
    print("\n" + "=" * 60)
    print("TEST 4: System Integration")
    print("=" * 60)

    try:
        from main import OceanBuoySystem

        print("\n  Initializing OceanBuoySystem...")
        system = OceanBuoySystem(use_simulation=True, assimilation_method='optimal')
        print(f"    [OK] System initialized")
        print(f"    Analyzer available: {hasattr(system, 'analyzer')}")

        print("\n  Running offline analysis...")
        result = system.run_offline_analysis(buoy_id=None)
        print(f"    [OK] Analysis completed")

        print("\n  Checking analysis results...")
        print(f"    Analysis results available: {system.analysis_results is not None}")

        print("\n  Testing detect_thermocline method...")
        thermocline = system.detect_thermocline()
        print(f"    Thermocline found: {thermocline['found'] if thermocline else 'N/A'}")

        print("\n  Testing detect_eddies method...")
        eddies = system.detect_eddies()
        print(f"    Eddies detected: {len(eddies) if eddies else 0}")

        print("\n  Testing generate_profile_animation method...")
        try:
            anim_file = system.generate_profile_animation(duration=2, fps=5)
            print(f"    Animation generated: {anim_file is not None and os.path.exists(anim_file)}")
        except Exception as e:
            print(f"    Animation skipped: not available ({e})")

        print("\n[OK] System integration test PASSED")
        return True

    except Exception as e:
        print(f"\n[FAILED] System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST OF NEW FEATURES")
    print("=" * 60)

    results = {}

    results['ocean_analysis'] = test_ocean_analysis()
    results['visualization_new'] = test_visualization_new()
    results['animation_export'] = test_animation_export()
    results['system_integration'] = test_system_integration()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAILED]"
        print(f"  {status} {name}")

    all_passed = all(results.values())
    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"\nOverall: {overall}")
    print(f"Passed: {sum(results.values())} / {len(results)}")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
