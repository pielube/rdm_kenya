import os
import sys
import time
import subprocess
import pandas as pd


def find_futures_root() -> str:
    # Resolve from repo root relative path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "1_Experiment", "Experimental_Platform", "Futures"))
    return root


def iter_cases(futures_root: str):
    # Yields (scenario_name, case_name, case_dir)
    if not os.path.isdir(futures_root):
        return
    for scen in sorted(os.listdir(futures_root)):
        scen_dir = os.path.join(futures_root, scen)
        if not os.path.isdir(scen_dir):
            continue
        for case in sorted(os.listdir(scen_dir)):
            case_dir = os.path.join(scen_dir, case)
            if not os.path.isdir(case_dir):
                continue
            yield scen, case, case_dir


def run_postprocess_for_case(case_name: str, scenario_name: str, solver: str = "cbc", tier: str = "3a", debug: bool = False) -> int:
    script = os.path.join(os.path.dirname(__file__), "create_csv_concatenate.py")
    cmd = [sys.executable, script, f"{case_name}.txt", tier, solver, "all", "--force", f"--scenario={scenario_name}"]
    if debug:
        cmd.append("--debug")
    return subprocess.call(cmd)


def aggregate_scenario_outputs(scen_dir: str, out_name: str = "Aggregated_Output.csv") -> str | None:
    rows = []
    for name in os.listdir(scen_dir):
        if name.endswith("_Output.csv"):
            case = name.replace("_Output.csv", "")
            p = os.path.join(scen_dir, name)
            try:
                df = pd.read_csv(p)
                df.insert(0, "CASE", case)
                rows.append(df)
            except Exception:
                continue
    if not rows:
        return None
    agg = pd.concat(rows, ignore_index=True, sort=False)
    out_path = os.path.join(scen_dir, out_name)
    agg.to_csv(out_path, index=False)
    return out_path


def main(argv):
    tier = "3a"
    solver = "cbc"
    do_aggregate = True
    if len(argv) > 1:
        tier = argv[1]
    if len(argv) > 2:
        solver = argv[2]
    if len(argv) > 3:
        do_aggregate = argv[3].lower() not in ("no", "false", "0")

    futures_root = find_futures_root()
    if not os.path.isdir(futures_root):
        print(f"Futures directory not found: {futures_root}")
        return 2

    print(f"Scanning Futures root: {futures_root}")
    processed = 0
    debug = os.environ.get('POSTPROC_DEBUG', '').lower() in ('1','true','yes')
    t_all = time.perf_counter()
    for scen, case, cdir in iter_cases(futures_root):
        input_parquet = os.path.join(cdir, f"{case}_Input.parquet")
        output_parquet = os.path.join(cdir, f"{case}_Output.parquet")
        if not (os.path.isfile(input_parquet) and os.path.isfile(output_parquet)):
            # Require both per-case parquet files
            continue
        t0 = time.perf_counter()
        rc = run_postprocess_for_case(case, scen, solver=solver, tier=tier, debug=debug)
        if debug:
            print(f"[RUNALL] {scen}/{case} finished rc={rc} in {time.perf_counter()-t0:.3f}s")
        if rc == 0:
            processed += 1
        else:
            print(f"Post-process failed for {scen}/{case} (exit {rc})")

    print(f"Post-processed cases: {processed} in {time.perf_counter()-t_all:.3f}s")

    if do_aggregate:
        # Aggregate per scenario
        for scen in sorted(os.listdir(futures_root)):
            scen_dir = os.path.join(futures_root, scen)
            if not os.path.isdir(scen_dir):
                continue
            out = aggregate_scenario_outputs(scen_dir)
            if out:
                print(f"Aggregated -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
