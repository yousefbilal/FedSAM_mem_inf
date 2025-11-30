import numpy as np
import glob
import os
import argparse


def _masked_avg(x, mask, axis=0, eps=1e-10):
    """Calculate masked average."""
    return (
        np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), eps)
    ).astype(np.float32)


def _masked_dot(x, mask, eps=1e-10):
    """Calculate masked dot product."""
    x = x.T.astype(np.float32)
    return (
        np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), eps)
    ).astype(np.float32)


def aggregate_results(results_dir, pattern):
    """
    Loads all .npz files matching the pattern, stacks their contents,
    and calculates final memorization and influence scores.
    """
    search_path = os.path.join(results_dir, pattern)
    result_files = glob.glob(search_path)

    if not result_files:
        print(
            f"No '.npz' files found in '{results_dir}' matching the pattern '{pattern}'"
        )
        return

    print(f"Found {len(result_files)} result files. Aggregating...")

    all_masks = []
    all_train_correctness = []
    all_test_correctness = []

    # Load data from all files
    for f_path in sorted(result_files):  # Sort to be tidy
        # print(f"Loading {f_path}...") # Uncomment for debugging
        try:
            data = np.load(f_path)
            all_masks.append(data["subset_mask"])
            all_train_correctness.append(data["train_correctness"])
            all_test_correctness.append(data["test_correctness"])
        except Exception as e:
            print(f"  Error loading {f_path}: {e}")

    if not all_masks:
        print("No data was successfully loaded.")
        return

    # Stack the lists of 1D arrays into 2D arrays
    trainset_mask = np.vstack(all_masks)
    trainset_correctness = np.vstack(all_train_correctness)
    testset_correctness = np.vstack(all_test_correctness)

    print("\n--- Aggregated Results ---")
    print(f"Total runs aggregated: {trainset_mask.shape[0]}")
    print(f"Train data points: {trainset_mask.shape[1]}")
    print(f"Test data points: {testset_correctness.shape[1]}")

    # --- Perform the final aggregation logic ---
    inv_mask = np.logical_not(trainset_mask)

    avg_test_acc = np.mean(testset_correctness)
    print(f"\nOverall Average test accuracy = {avg_test_acc:.4f}")

    mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(
        trainset_correctness, inv_mask
    )
    infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(
        testset_correctness, inv_mask
    )

    print(f"Overall Memorization (shape): {mem_est.shape}")
    print(f"Overall Influence (shape): {infl_est.shape}")

    # Save the final aggregated results
    final_save_path = os.path.join(results_dir, "AGGREGATED_RESULTS_FINAL.npz")
    np.savez_compressed(
        final_save_path,
        memorization=mem_est,
        influence=infl_est,
        avg_test_acc=avg_test_acc,
        total_runs=trainset_mask.shape[0],
    )
    print(f"\nFinal aggregated results saved to {final_save_path}")
    return final_save_path


parser = argparse.ArgumentParser(description="aggregate results.")
parser.add_argument("--results-dir", required=True)
args = parser.parse_args()
pattern = "*_correctness*.npz"

aggregate_results(args.results_dir, pattern)
