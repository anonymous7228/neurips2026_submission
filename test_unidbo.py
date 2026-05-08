import os
import glob
import pickle
import torch
import csv
import argparse
import numpy as np
import datetime
import time
import yaml
try:
    import mediapy  # type: ignore
except Exception:
    mediapy = None
import cv2
from tqdm import tqdm

# Ensure matplotlib uses a writable cache/config directory in restricted envs.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

# set tf to cpu only
import tensorflow as tf
import jax

# utils
from unidbo_core.data.dataset import WaymaxTestDataset
from unidbo_core.model.utils import set_seed
from unidbo_core.sim_agent.utils import sample_to_action
from unidbo_core.waymax_visualization.plotting import plot_state

# waymax
from waymax import dynamics
from waymax import datatypes
from waymax import env as _env
from waymax import visualization
from waymax.config import EnvironmentConfig, ObjectType
from waymax.config import DatasetConfig, DataFormat
from waymax.metrics.comfort import KinematicsInfeasibilityMetric
from waymax.metrics import OverlapMetric, LogDivergenceMetric
from unidbo_core.sim_agent.waymax_metrics import OffroadMetric, WrongWayMetric
from unidbo_core.sim_agent.waymax_env import WaymaxEnvironment
from unidbo_core.data.waymax_utils import create_iter


## Parameters
CURRENT_TIME_INDEX = 10
N_SIM_AGENTS = 32
N_SIMULATION_STEPS = 80


## Set up Waynax Environment
env_config = EnvironmentConfig(
    # Ensure that the sim agent can control all valid objects.
    controlled_object=ObjectType.VALID,
    max_num_objects=N_SIM_AGENTS,
    allow_new_objects_after_warmup=False
)

dynamics_model = dynamics.StateDynamics()

env = WaymaxEnvironment(
    dynamics_model=dynamics_model,
    config=env_config,
)

dataset = None


## Calculate metrics
def calculate_metrics(metrics, modeled_indices):
    if not modeled_indices:
        return {
            'collision': 0.0,
            'offroad': 0.0,
            'wrong_way': 0.0,
            'log_divergence': 0.0,
            'kinematic_infeasibility': 0.0,
        }

    offroad = []
    collision = []
    log_divergence = []
    wrong_way = []
    kinematic_infeasibility = []

    for i in modeled_indices:
        is_offroad = bool(metrics[0]['offroad'].value[i].item())
        is_collision = bool(metrics[0]['overlap'].value[i].item())
        col, off, kin, div, wrw = [], [], [], [], []

        for t in range(len(metrics)):
            valid = metrics[t]['log_divergence'].valid[i]
            div.append((metrics[t]['log_divergence'].value[i] * valid).item())
            col.append(bool(metrics[t]['overlap'].value[i].item()))
            off.append(bool(metrics[t]['offroad'].value[i].item()))
            kin.append((metrics[t]['kinematic_infeasibility'].value[i]).item())
            wrw.append(float(metrics[t]['wrong_way'].value[i].item()))

        col_np = np.asarray(col, dtype=np.bool_)
        off_np = np.asarray(off, dtype=np.bool_)
        wrw_np = np.asarray(wrw, dtype=np.float32)
        div_np = np.asarray(div, dtype=np.float32)
        kin_np = np.asarray(kin, dtype=np.float32)

        collision_eval = bool(np.any(col_np)) if not is_collision else False
        offroad_eval = bool(np.any(off_np)) if not is_offroad else False
        wrong_way_eval = bool(np.sum(wrw_np) > 10)

        collision.append(collision_eval)
        offroad.append(offroad_eval)
        wrong_way.append(wrong_way_eval)
        log_divergence.append(float(np.mean(div_np)))
        kinematic_infeasibility.append(float(np.mean(kin_np)))

    return {
        'collision': np.mean(collision),
        'offroad': np.mean(offroad),
        'wrong_way': np.mean(wrong_way),
        'log_divergence': np.mean(log_divergence),
        'kinematic_infeasibility': np.mean(kinematic_infeasibility),
    }


def _compute_ever_collision_mask(log_metrics):
    """Builds per-agent mask for collisions that happened during rollout.

    Matches metrics semantics by excluding agents already collided at the first
    evaluated step.
    """
    if not log_metrics:
        return None

    first_overlap = np.asarray(log_metrics[0]['overlap'].value).astype(bool).reshape(-1)
    ever_overlap = first_overlap.copy()
    for item in log_metrics[1:]:
        curr = np.asarray(item['overlap'].value).astype(bool).reshape(-1)
        if curr.shape[0] < ever_overlap.shape[0]:
            padded = np.zeros_like(ever_overlap, dtype=bool)
            padded[:curr.shape[0]] = curr
            curr = padded
        elif curr.shape[0] > ever_overlap.shape[0]:
            curr = curr[:ever_overlap.shape[0]]
        ever_overlap |= curr
    return ever_overlap & (~first_overlap)


def _compute_ever_collision_mask_no_baseline_exclusion(log_metrics):
    """Builds per-agent ever-collision mask across rollout without exclusion.

    This is used for visualization-only persistent highlighting:
    once collided at any timestep, the agent stays red in mp4/png.
    """
    if not log_metrics:
        return None

    ever_overlap = None
    for item in log_metrics:
        curr = np.asarray(item['overlap'].value).astype(bool).reshape(-1)
        if ever_overlap is None:
            ever_overlap = curr.copy()
            continue
        if curr.shape[0] < ever_overlap.shape[0]:
            padded = np.zeros_like(ever_overlap, dtype=bool)
            padded[:curr.shape[0]] = curr
            curr = padded
        elif curr.shape[0] > ever_overlap.shape[0]:
            curr = curr[:ever_overlap.shape[0]]
        ever_overlap |= curr
    return ever_overlap


## Begin Simulation
def _list_tfrecord_files(test_path: str):
    if test_path is None:
        return []
    if not os.path.exists(test_path) and not ("*" in test_path or "?" in test_path or "[" in test_path):
        return []
    if os.path.isdir(test_path):
        files = sorted(glob.glob(os.path.join(test_path, "uncompressed_tf_example_validation_validation_tfexample.tfrecord*")))
        if files:
            return files
        return sorted(glob.glob(os.path.join(test_path, "*.tfrecord*")))
    if "*" in test_path or "?" in test_path or "[" in test_path:
        return sorted(glob.glob(test_path))
    return [test_path]


def _configure_runtime_auto():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus, 'GPU')
        except RuntimeError:
            pass
        try:
            jax.devices("gpu")
            jax.config.update('jax_platform_name', 'gpu')
        except Exception:
            jax.config.update('jax_platform_name', 'cpu')
    else:
        tf.config.set_visible_devices([], 'GPU')
        jax.config.update('jax_platform_name', 'cpu')


def _resolve_unidbo_cfg_from_checkpoint(model_path: str, model_dim_override: int = None) -> dict:
    """
    Resolve UniDBO init cfg from checkpoint hyperparameters,
    and optionally override model dimension from CLI.
    """
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {}) if isinstance(ckpt, dict) else {}

    cfg = None
    if isinstance(hparams, dict):
        cfg = hparams.get('cfg', None)
        if cfg is None and {'future_len', 'agents_len', 'action_len'}.issubset(set(hparams.keys())):
            # fallback: some checkpoints may flatten cfg into hyper_parameters
            cfg = dict(hparams)

    if not isinstance(cfg, dict):
        raise RuntimeError(
            "Checkpoint does not contain a valid `hyper_parameters.cfg`; "
            "cannot build UniDBO reliably."
        )

    cfg = dict(cfg)
    if model_dim_override is not None:
        if int(model_dim_override) <= 0:
            raise ValueError(f"Invalid --model_dim_override={model_dim_override}, expected > 0")
        cfg['model_dim'] = int(model_dim_override)

    return cfg


def _reset_agent_length_compat(model, agents_len: int) -> None:
    """
    Compatibility helper for models without `reset_agent_length` method.
    """
    if hasattr(model, 'reset_agent_length'):
        model.reset_agent_length(agents_len)
        return

    setattr(model, '_agents_len', int(agents_len))
    branch = getattr(model, "dhn_branch", None)
    if branch is not None and hasattr(branch, 'reset_agent_length'):
        branch.reset_agent_length(agents_len)


def _sample_unidbo_dhn(model, batch: dict, args):
    return model.sample_closed_loop(batch, terminal_step=args.dhn_terminal_step)


def _format_elapsed_time(seconds: float) -> str:
    total = max(float(seconds), 0.0)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60.0
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    return f"{secs:.2f}s"


def _sync_torch_if_cuda(device) -> None:
    """Synchronize CUDA kernels for accurate latency measurement."""
    if not torch.cuda.is_available():
        return
    if device is None:
        return
    dev = str(device).lower()
    if dev.startswith('cuda'):
        torch.cuda.synchronize()


def _sync_jax_if_needed(obj) -> None:
    """Synchronize JAX execution for accurate runtime measurement."""
    try:
        jax.block_until_ready(obj)
    except Exception:
        pass


def _normalize_scenario_id(sid) -> str:
    if isinstance(sid, bytes):
        return sid.decode('utf-8')
    return str(sid)


def _load_target_scenario_ids(args) -> set:
    ids = set()
    if getattr(args, "scenario_ids", None):
        for token in str(args.scenario_ids).split(","):
            token = token.strip()
            if token:
                ids.add(token)

    ids_file = getattr(args, "scenario_ids_file", None)
    if ids_file:
        if not os.path.exists(ids_file):
            raise FileNotFoundError(f"scenario_ids_file not found: {ids_file}")
        if ids_file.endswith(".csv"):
            with open(ids_file, newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise RuntimeError(f"Invalid CSV (no header): {ids_file}")
                if "scenario_id" in reader.fieldnames:
                    for row in reader:
                        sid = str(row.get("scenario_id", "")).strip()
                        if sid:
                            ids.add(sid)
                else:
                    # fallback: use first column
                    first_col = reader.fieldnames[0]
                    for row in reader:
                        sid = str(row.get(first_col, "")).strip()
                        if sid:
                            ids.add(sid)
        else:
            with open(ids_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sid = line.strip()
                    if sid:
                        ids.add(sid)
    return ids


def _write_video_compat(mp4_path: str, frames: list, fps: int = 10) -> None:
    if not frames:
        raise RuntimeError("No frames to write.")
    if mediapy is not None:
        try:
            mediapy.write_video(mp4_path, frames, fps=fps)
            return
        except Exception:
            pass

    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] != 3:
        raise RuntimeError(f"Invalid frame shape for video: {first.shape}")
    h, w = int(first.shape[0]), int(first.shape[1])
    writer = cv2.VideoWriter(
        mp4_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")
    for frame in frames:
        img = np.asarray(frame)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        # RGB -> BGR
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()


def run_simulation(args):
    run_start_time = time.perf_counter()
    def _print_total_runtime():
        elapsed = time.perf_counter() - run_start_time
        print(f"Total closed-loop inference runtime: {_format_elapsed_time(elapsed)} ({elapsed:.3f}s)")

    global dataset
    _configure_runtime_auto()
    if dataset is None:
        dataset = WaymaxTestDataset(
            data_dir=None,
            max_object=N_SIM_AGENTS
        )
    target_scenario_ids = _load_target_scenario_ids(args)
    if target_scenario_ids:
        print(f"Scenario filtering enabled: {len(target_scenario_ids)} target scenarios")
    processed_target_ids = set()
    ## Load model
    print(f"Loading model: {args.model_path}")
    try:
        resolved_cfg = _resolve_unidbo_cfg_from_checkpoint(
            args.model_path,
            model_dim_override=args.model_dim_override,
        )
        effective_model_dim = int(resolved_cfg.get('model_dim', 256))
        if 'diffusion_steps' not in resolved_cfg:
            raise RuntimeError(
                "This script only supports UniDBO diffusion checkpoints (requires diffusion_steps)."
            )
        if args.model_dim_override is not None:
            print(f"Model dimension override: model_dim={effective_model_dim} (from CLI)")
        else:
            print(f"Model dimension: model_dim={effective_model_dim} (from checkpoint)")

        from unidbo_model import UniDBOModel
        print("Model type: UniDBO (DHN)")
        model = UniDBOModel.load_from_checkpoint(
            args.model_path,
            map_location=args.device,
            strict=False,
            cfg=resolved_cfg,
        )

        _reset_agent_length_compat(model, N_SIM_AGENTS)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        _print_total_runtime()
        return
    
    set_seed(args.seed)  
    
    # Load testing scenarios
    file_list = _list_tfrecord_files(args.test_path)
    if not file_list:
        print(f"Test data not found: {args.test_path}")
        _print_total_runtime()
        return
    print(f"Loading test data: {args.test_path}")
    print(f"Discovered {len(file_list)} TFRecord files")
    print(
        "DHN closed-loop config: "
        f"terminal_step={args.dhn_terminal_step} "
        "(<0 means auto-use diffusion_steps-1)"
    )

    # Save mode:
    # - csv: only save metrics.csv
    # - all: save metrics.csv + scenario mp4 + scenario pkl
    save_mode = getattr(args, "save_mode", "csv")
    if getattr(args, "save_simulation", False):
        # backward compatibility for existing commands
        save_mode = "all"
    save_full_outputs = (save_mode == "all")

    # Save results: create a new output folder per run.
    if getattr(args, "output_dir", None):
        SAVE_PATH = os.path.abspath(args.output_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        SAVE_PATH = f'testing_results/test_dhn/{args.seed}_{timestamp}'
    print(f"Outputs will be saved to: {SAVE_PATH}")
    print(f"Save mode: {save_mode} (csv{' + mp4 + pkl' if save_full_outputs else ''})")
    
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Output directory created: {SAVE_PATH}")
    except Exception as e:
        print(f"Failed to create output directory: {e}")
        _print_total_runtime()
        return
    

    metrics_file = os.path.join(SAVE_PATH, 'metrics.csv')
    metric_sums = {
        'collision': 0.0,
        'offroad': 0.0,
        'wrong_way': 0.0,
        'log_divergence': 0.0,
        'kinematic_infeasibility': 0.0,
    }
    metric_count = 0

    # Waymax-style runtime accounting (ms per simulated step).
    reset_total_s = 0.0
    model_infer_total_s = 0.0
    transition_total_s = 0.0
    metrics_total_s = 0.0
    sim_steps_total = 0
    

    with open(metrics_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['scenario_id', 'collision', 'offroad', 'wrong_way', 
                        'log_divergence', 'kinematic_infeasibility', 'active_count', 'total_valid'])
    print(f"Metrics file initialized: {metrics_file}")

    # Begin simulation
    sim_loop_start_t = time.perf_counter()
    scenario_count = 0
    for tfrecord_path in file_list:
        if args.max_scenarios is not None and scenario_count >= args.max_scenarios:
            print(f"Reached max scenario limit: {args.max_scenarios}")
            break

        print(f"Processing file: {tfrecord_path}")
        config = DatasetConfig(
            path=tfrecord_path,
            max_num_objects=N_SIM_AGENTS,
            repeat=1,
            max_num_rg_points=30000,
            data_format=DataFormat.TFRECORD
        )

        try:
            data_iter = create_iter(config)
        except Exception as e:
            print(f"Failed to create data iterator: {e}")
            continue

        pbar = tqdm(data_iter, desc="Sim")
        for scenario_id, scenario in pbar:
            scenario_id = _normalize_scenario_id(scenario_id)
            if args.max_scenarios is not None and scenario_count >= args.max_scenarios:
                print(f"Reached max scenario limit: {args.max_scenarios}")
                break
            if target_scenario_ids and scenario_id not in target_scenario_ids:
                continue

            scenario_count += 1
            pbar.set_postfix_str(f"{scenario_id}")
            print(f"Running scenario {scenario_id} ({scenario_count})...")

            _sync_torch_if_cuda(model.device)
            reset_start_t = time.perf_counter()
            initial_state = current_state = env.reset(scenario)
            _sync_jax_if_needed(current_state)
            _sync_torch_if_cuda(model.device)
            reset_end_t = time.perf_counter()
            reset_total_s += (reset_end_t - reset_start_t)
            log_states = [initial_state]
            log_metrics = []    
            is_valid = scenario.object_metadata.is_valid
            is_controlled = is_valid[:N_SIM_AGENTS]

            if args.active_only:
                log_traj = scenario.log_trajectory
                vel_x = np.array(log_traj.vel_x)
                vel_y = np.array(log_traj.vel_y)
                speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
                non_static = np.any(speed > 0.1, axis=1)
                active_vehicles = np.array(is_controlled) & non_static[:len(is_controlled)]
                modeled_indices = np.where(active_vehicles)[0].tolist()
            else:
                modeled_indices = jax.numpy.where(is_controlled)[0].tolist()
            total_controlled = int(np.sum(is_controlled))
            active_count = len(modeled_indices)
            print(
                f"Evaluation objects: active={active_count}, total_valid={total_controlled}, active_only={args.active_only}"
            )

            # Run the simulated scenarios.
            for t in (range(initial_state.remaining_timesteps)):
                i = t % args.replan

                if i == 0:
                    print("Replan at ", current_state.timestep)

                    with torch.no_grad():
                        sample = dataset.process_scenario(current_state, current_state.timestep, use_log=False)
                        batch = dataset.__collate_fn__([sample])

                        _sync_torch_if_cuda(model.device)
                        model_start_t = time.perf_counter()
                        pred = _sample_unidbo_dhn(
                            model=model,
                            batch=batch,
                            args=args,
                        )
                        pred_traj = pred['denoised_trajs'].cpu().numpy()[0]
                        _sync_torch_if_cuda(model.device)
                        model_end_t = time.perf_counter()

                        model_infer_total_s += (model_end_t - model_start_t)

                sample = pred_traj[:, i, :]
                action = sample_to_action(sample, is_controlled, None, N_SIM_AGENTS)

                _sync_torch_if_cuda(model.device)
                transition_start_t = time.perf_counter()
                current_state = env.step_sim_agent(current_state, [action])
                _sync_jax_if_needed(current_state)
                _sync_torch_if_cuda(model.device)
                transition_end_t = time.perf_counter()
                transition_total_s += (transition_end_t - transition_start_t)
                log_states.append(current_state)

                # Run metrics
                _sync_torch_if_cuda(model.device)
                metrics_start_t = time.perf_counter()
                overlap = OverlapMetric().compute(current_state)
                offroad = OffroadMetric().compute(current_state)
                wrongway = WrongWayMetric().compute(current_state)
                log_divergence = LogDivergenceMetric().compute(current_state)
                kinematic_infeasibility = KinematicsInfeasibilityMetric().compute(current_state)
                _sync_jax_if_needed(
                    (
                        overlap,
                        offroad,
                        wrongway,
                        log_divergence,
                        kinematic_infeasibility,
                    )
                )
                _sync_torch_if_cuda(model.device)
                metrics_end_t = time.perf_counter()
                metrics_total_s += (metrics_end_t - metrics_start_t)
                log_metrics.append({
                    'overlap': overlap,
                    'offroad': offroad,
                    'wrong_way': wrongway,
                    'log_divergence': log_divergence,
                    'kinematic_infeasibility': kinematic_infeasibility
                })
                sim_steps_total += 1
            

            # Calculate metrics
            metrics = calculate_metrics(log_metrics, modeled_indices)
            with open(metrics_file, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([scenario_id, metrics['collision'], metrics['offroad'], metrics['wrong_way'], 
                                metrics['log_divergence'], metrics['kinematic_infeasibility'], active_count, total_controlled])
            print(f"Saved metrics for scenario {scenario_id}")
            for key in metric_sums:
                metric_sums[key] += metrics[key]
            metric_count += 1

            # Build per-agent persistent collision mask for visualization.
            ever_collided_mask_vis = _compute_ever_collision_mask_no_baseline_exclusion(log_metrics)

            # Optional artifact outputs: mp4 + pkl
            save_mp4_pkl = save_full_outputs
            if save_mp4_pkl:
                sim_images = []
                for state in log_states:
                    img = plot_state(
                        state,
                        show_agent_ids=bool(getattr(args, "show_agent_ids", False)),
                        agent_box_scale=float(getattr(args, "agent_box_scale", 1.35)),
                        force_overlap_mask=ever_collided_mask_vis,
                    )
                    sim_images.append(img)

                with open(os.path.join(SAVE_PATH, f'{scenario_id}_sim.pkl'), 'wb') as f:
                    pickle.dump(log_states[-1].sim_trajectory, f)

                mp4_path = os.path.join(SAVE_PATH, f'{scenario_id}.mp4')
                try:
                    _write_video_compat(mp4_path, sim_images, fps=10)
                    print(f"Saved simulation video for scenario {scenario_id}: {mp4_path}")
                except Exception as e:
                    print(f"Failed to save simulation video for scenario {scenario_id}: {e}")

            if target_scenario_ids:
                processed_target_ids.add(scenario_id)
                if bool(getattr(args, "stop_when_all_targets_found", True)) and processed_target_ids >= target_scenario_ids:
                    print("All target scenarios processed. Exiting early.")
                    break
        if target_scenario_ids and bool(getattr(args, "stop_when_all_targets_found", True)) and processed_target_ids >= target_scenario_ids:
            break

    print(f"Simulation finished. Total processed scenarios: {scenario_count}")
    print(f"All outputs saved in: {SAVE_PATH}")
    if target_scenario_ids:
        missed = sorted(list(target_scenario_ids - processed_target_ids))
        print(f"Target scenario coverage: {len(processed_target_ids)}/{len(target_scenario_ids)}")
        if missed:
            print("Missed scenario IDs:", ",".join(missed))
    if metric_count > 0:
        print("\n--- Average Evaluation Metrics ---")
        for key, total in metric_sums.items():
            print(f"{key}: {total / metric_count:.6f}")
        print("--------------------")

    sim_loop_total_s = time.perf_counter() - sim_loop_start_t
    if sim_steps_total > 0 and scenario_count > 0:
        sim_time_component_s_per_scene = (
            reset_total_s + model_infer_total_s + transition_total_s + metrics_total_s
        ) / float(scenario_count)
        sim_time_wall_s_per_scene = sim_loop_total_s / float(scenario_count)
        reset_ms_per_scene = 1000.0 * reset_total_s / float(scenario_count)
        infer_ms_per_step = 1000.0 * model_infer_total_s / float(sim_steps_total)
        transition_ms_per_step = 1000.0 * transition_total_s / float(sim_steps_total)
        metrics_ms_per_step = 1000.0 * metrics_total_s / float(sim_steps_total)
        step_env_ms_per_step = transition_ms_per_step + metrics_ms_per_step
        total_ms_per_step = infer_ms_per_step + transition_ms_per_step + metrics_ms_per_step
        total_hz = 1000.0 / total_ms_per_step if total_ms_per_step > 0 else 0.0

        print("\n--- Waymax-Style Runtime Statistics ---")
        print(f"Sim Time (component-sum): {sim_time_component_s_per_scene:.6f} s/scene")
        print(f"Sim Time (wall-clock loop): {sim_time_wall_s_per_scene:.6f} s/scene")
        print(f"Reset: {reset_ms_per_scene:.6f} ms/scene")
        print(f"Inference(amortized): {infer_ms_per_step:.6f} ms/step")
        print(f"Transition: {transition_ms_per_step:.6f} ms/step")
        print(f"Metrics: {metrics_ms_per_step:.6f} ms/step")
        print(f"Step (Transition+Metrics): {step_env_ms_per_step:.6f} ms/step")
        print(f"Total (Inference+Transition+Metrics): {total_ms_per_step:.6f} ms/step")
        print(f"Total Throughput: {total_hz:.3f} Hz")
        print("---------------------------")
    _print_total_runtime()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/closed_loop.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument(
        '--model_dim_override',
        type=int,
        default=None,
        help='Override model backbone dim for inference (e.g., 128 or 256). Default: use checkpoint cfg',
    )
    parser.add_argument('--replan', type=int, default=10, help='Replan frequency')
    parser.add_argument('--dhn_terminal_step', type=int, default=-1, help='DHN terminal diffusion index; -1 means diffusion_steps-1')
    parser.add_argument(
        '--save_mode',
        type=str,
        choices=['csv', 'all'],
        default='csv',
        help='Save mode: csv (metrics only) or all (metrics + mp4 + pkl)',
    )
    parser.add_argument(
        '--save_simulation',
        action='store_true',
        help='Deprecated compatibility flag, equivalent to --save_mode all',
    )
    parser.add_argument('--max_scenarios', type=int, default=None, help='Maximum number of scenarios to run')
    parser.add_argument('--active_only', action=argparse.BooleanOptionalAction, default=True, help='Only evaluate active (non-static) vehicles')
    parser.add_argument('--scenario_ids', type=str, default=None, help='Comma-separated scenario IDs to run')
    parser.add_argument('--scenario_ids_file', type=str, default=None, help='Path to txt/csv containing scenario IDs (csv preferred with scenario_id column)')
    parser.add_argument('--stop_when_all_targets_found', action=argparse.BooleanOptionalAction, default=True, help='Stop early when all target scenario_ids are processed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory override')
    parser.add_argument('--show_agent_ids', action=argparse.BooleanOptionalAction, default=False, help='Show numeric agent IDs on vehicles in visualizations')
    parser.add_argument('--agent_box_scale', type=float, default=1.35, help='Scale factor for rendered vehicle boxes (>=1 makes cars larger)')

    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg_args = yaml.safe_load(f) or {}
        defaults = {action.dest: action.default for action in parser._actions if action.dest != 'help'}
        for key, value in cfg_args.items():
            if hasattr(args, key) and getattr(args, key) == defaults.get(key):
                setattr(args, key, value)
    run_simulation(args)
