import argparse
import glob
import os
import pickle
import sys

from tqdm import tqdm
from waymax.config import DataFormat, DatasetConfig

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from unidbo_core.data.data_utils import data_process_scenario
from unidbo_core.data.waymax_utils import create_iter


def list_tfrecord_files(input_path: str):
    if os.path.isdir(input_path):
        files = sorted(
            glob.glob(
                os.path.join(
                    input_path,
                    "uncompressed_tf_example_validation_validation_tfexample.tfrecord*",
                )
            )
        )
        if files:
            return files
        return sorted(glob.glob(os.path.join(input_path, "*.tfrecord*")))
    if "*" in input_path or "?" in input_path or "[" in input_path:
        return sorted(glob.glob(input_path))
    if os.path.isfile(input_path):
        return [input_path]
    return []


def list_pickle_files(input_path: str):
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.pkl")))
    if "*" in input_path or "?" in input_path or "[" in input_path:
        return sorted([p for p in glob.glob(input_path) if p.endswith(".pkl")])
    if os.path.isfile(input_path) and input_path.endswith(".pkl"):
        return [input_path]
    return []


def convert_one_file(tfrecord_path: str, args, counters: dict):
    cfg = DatasetConfig(
        path=tfrecord_path,
        data_format=DataFormat.TFRECORD,
        repeat=1,
        max_num_rg_points=int(args.max_num_rg_points),
        max_num_objects=int(args.max_num_objects),
    )

    data_iter = create_iter(cfg)
    pbar = tqdm(data_iter, desc=f"Convert {os.path.basename(tfrecord_path)}")
    for scenario_id, scenario in pbar:
        if args.max_scenarios is not None and counters["saved"] >= int(args.max_scenarios):
            return

        out_name = f"scenario_{scenario_id}.pkl"
        out_path = os.path.join(args.output_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            counters["skipped"] += 1
            continue

        data_dict = data_process_scenario(
            scenario=scenario,
            max_num_objects=int(args.max_num_objects),
            max_polylines=int(args.max_polylines),
            current_index=int(args.current_index),
            num_points_polyline=int(args.num_points_polyline),
            use_log=bool(args.use_log),
            selected_agents=None,
            remove_history=False,
        )
        data_dict["scenario_id"] = str(scenario_id)
        if args.save_raw:
            data_dict["scenario_raw"] = scenario

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, out_path)
        counters["saved"] += 1


def convert_pickles(pickle_files, args, counters: dict):
    pbar = tqdm(pickle_files, desc="Convert pickle")
    for pkl_path in pbar:
        if args.max_scenarios is not None and counters["saved"] >= int(args.max_scenarios):
            return

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            scenario = data.get("scenario_raw", data.get("scenario", None))
            scenario_id = data.get("scenario_id", None)
        else:
            scenario = None
            scenario_id = None

        if scenario is None:
            counters["skipped"] += 1
            continue

        if scenario_id is None:
            scenario_id = os.path.basename(pkl_path).split(".")[0].replace("scenario_", "")
        scenario_id = str(scenario_id)

        out_name = f"scenario_{scenario_id}.pkl"
        out_path = os.path.join(args.output_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            counters["skipped"] += 1
            continue

        data_dict = data_process_scenario(
            scenario=scenario,
            max_num_objects=int(args.max_num_objects),
            max_polylines=int(args.max_polylines),
            current_index=int(args.current_index),
            num_points_polyline=int(args.num_points_polyline),
            use_log=bool(args.use_log),
            selected_agents=None,
            remove_history=False,
        )
        data_dict["scenario_id"] = scenario_id
        if args.save_raw:
            data_dict["scenario_raw"] = scenario

        tmp_path = out_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, out_path)
        counters["saved"] += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Raw WOMD/Waymax tfrecord file, directory, or glob pattern.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store converted scenario_*.pkl files.",
    )
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--max_num_objects", type=int, default=64)
    parser.add_argument("--max_polylines", type=int, default=256)
    parser.add_argument("--num_points_polyline", type=int, default=30)
    parser.add_argument("--current_index", type=int, default=10, help="History length - 1 index.")
    parser.add_argument("--max_num_rg_points", type=int, default=30000)
    parser.add_argument("--use_log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_raw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    counters = {"saved": 0, "skipped": 0}

    tfrecord_files = list_tfrecord_files(args.input_path)
    pickle_files = list_pickle_files(args.input_path) if not tfrecord_files else []
    if not tfrecord_files and not pickle_files:
        raise FileNotFoundError(
            f"No supported input files found from input_path={args.input_path}. "
            "Expect tfrecord* or .pkl files."
        )

    if tfrecord_files:
        print(f"Found {len(tfrecord_files)} tfrecord file(s).")
        for path in tfrecord_files:
            if args.max_scenarios is not None and counters["saved"] >= int(args.max_scenarios):
                break
            convert_one_file(path, args, counters)
    else:
        print(f"Found {len(pickle_files)} pickle file(s).")
        convert_pickles(pickle_files, args, counters)

    print(
        f"Done. saved={counters['saved']}, skipped_existing={counters['skipped']}, "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
