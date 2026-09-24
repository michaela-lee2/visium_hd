"""
Run the full Visium pipeline: QC -> Clustering/DEG -> cell2location Deconvolution.

Each step runs in its own subprocess so that peak memory usage can be
monitored and logged over time (useful for large Visium HD datasets).

Usage
-----
python pipeline.py \
    --visium-dir /path/to/sample/ \
    --count-file filtered_feature_bc_matrix.h5 \
    --reference-csv /path/to/reference_signatures.csv \
    --output-prefix my_sample \
    --resolution 0.5 \
    --n-top-genes 2000 \
    --n-cells-per-location 10 \
    --max-epochs 20000 \
    --batch-size 2400
"""

import argparse
import multiprocessing
import time
from functools import partial

import pandas as pd
import psutil

from qc import run_qc_pipeline
from clustering_deg import clustering_deg
from deconvolution import run_cell2location_deconvolution


def target_wrapper(queue, func, *args, **kwargs):
    """Run `func` and push its return value onto `queue` (for use in a subprocess)."""
    result = func(*args, **kwargs)
    queue.put(result)


def run_and_monitor(target_func, args_dict, log_list, event_name, pipeline_start_time, interval=5):
    """
    Run `target_func` in a subprocess while periodically logging its RSS memory usage.

    Parameters
    ----------
    target_func
        The function to run (e.g. run_qc_pipeline).
    args_dict
        Keyword arguments passed to `target_func`.
    log_list
        List that memory/time log entries are appended to.
    event_name
        Label for this pipeline step, used in the log (e.g. "QC").
    pipeline_start_time
        `time.time()` value marking the start of the whole pipeline.
    interval
        Seconds between memory samples.
    """
    q = multiprocessing.Queue()
    wrapped_func = partial(target_wrapper, q, target_func, **args_dict)
    p = multiprocessing.Process(target=wrapped_func)

    print(f"\n--- Step: [{event_name}] started (PID: {p.pid}) ---")
    p.start()

    ps_process = psutil.Process(p.pid)
    mem_usage = 0.0
    while p.is_alive():
        try:
            mem_usage = ps_process.memory_info().rss / (1024 * 1024)  # MB
            elapsed_time = time.time() - pipeline_start_time
            log_list.append(
                {"Elapsed_Time_sec": elapsed_time, "Memory_MB": mem_usage, "Event": f"Running_{event_name}"}
            )
            time.sleep(interval)
        except psutil.NoSuchProcess:
            break

    p.join()
    result = q.get()

    final_mem = ps_process.memory_info().rss / (1024 * 1024) if ps_process.is_running() else mem_usage
    elapsed_time = time.time() - pipeline_start_time
    log_list.append({"Elapsed_Time_sec": elapsed_time, "Memory_MB": final_mem, "Event": f"End_{event_name}"})

    print(f"--- Step: [{event_name}] finished ---")
    return result


def format_duration(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Visium QC -> Clustering/DEG -> Deconvolution pipeline.")
    parser.add_argument("--visium-dir", required=True, help="Directory containing the Visium sample data.")
    parser.add_argument("--count-file", required=True, help="Name of the .h5 count matrix file.")
    parser.add_argument("--reference-csv", required=True, help="Path to reference cell-type signature CSV.")
    parser.add_argument("--output-prefix", default="visium_run", help="Prefix for all output files.")
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of highly variable genes.")
    parser.add_argument("--resolution", type=float, default=0.5, help="Leiden clustering resolution.")
    parser.add_argument("--n-cells-per-location", type=int, default=10, help="cell2location N_cells_per_location prior.")
    parser.add_argument("--max-epochs", type=int, default=20000, help="cell2location training epochs.")
    parser.add_argument("--batch-size", type=int, default=2400, help="cell2location training batch size.")
    parser.add_argument("--monitor-interval", type=int, default=5, help="Seconds between memory usage samples.")
    return parser.parse_args()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()

    performance_log = []
    pipeline_start_time = time.time()

    print("=" * 50)
    print("Starting full pipeline run")
    print("=" * 50)

    # --- Step 1: QC ---
    adata_qc = run_and_monitor(
        target_func=run_qc_pipeline,
        args_dict={"path": args.visium_dir, "count_file": args.count_file},
        log_list=performance_log,
        event_name="QC",
        pipeline_start_time=pipeline_start_time,
        interval=args.monitor_interval,
    )

    # --- Step 2: Clustering + DEG ---
    processed_adata = run_and_monitor(
        target_func=clustering_deg,
        args_dict={
            "adata": adata_qc.copy(),
            "n_top_genes": args.n_top_genes,
            "resolution": args.resolution,
            "output_prefix": args.output_prefix,
        },
        log_list=performance_log,
        event_name="Clustering_DEG",
        pipeline_start_time=pipeline_start_time,
        interval=args.monitor_interval,
    )

    # --- Step 3: Deconvolution ---
    adata_deconv = run_and_monitor(
        target_func=run_cell2location_deconvolution,
        args_dict={
            "adata": adata_qc,
            "reference_csv": args.reference_csv,
            "n_cells_per_location": args.n_cells_per_location,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
        },
        log_list=performance_log,
        event_name="Deconvolution",
        pipeline_start_time=pipeline_start_time,
        interval=args.monitor_interval,
    )

    # --- Merge results and save ---
    print("\n--- Merging results and saving output files ---")
    obs_csv = f"{args.output_prefix}_deconv_obs.csv"
    obs_df = adata_deconv.obs.copy()
    obs_df.index.name = "barcode"
    obs_df.to_csv(obs_csv, encoding="utf-8-sig")

    processed_adata.obs = processed_adata.obs.merge(obs_df, left_index=True, right_on="barcode", how="left")
    processed_adata.write_h5ad(f"{args.output_prefix}_processed.h5ad")
    print("--- Merge and save complete ---\n")

    # --- Save performance log ---
    log_df = pd.DataFrame(performance_log)
    log_csv = f"{args.output_prefix}_pipeline_performance_log.csv"
    log_df.to_csv(log_csv, index=False)

    print("=" * 50)
    print(f"Performance log saved to '{log_csv}'")
    print(log_df.head())
    print("...")
    print(log_df.tail())
    print("=" * 50)

    pipeline_duration = time.time() - pipeline_start_time
    final_mem = log_df["Memory_MB"].max()

    print("\n" + "=" * 50)
    print("Pipeline completed successfully.")
    print(f"Total runtime: {format_duration(pipeline_duration)}")
    print(f"Peak memory usage: {final_mem:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()
