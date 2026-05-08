import unidbo_core.waymax_visualization as visualization
import numpy as np
from matplotlib import pyplot as plt
try:
    import mediapy  # type: ignore
except Exception:
    mediapy = None
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None


def _resize_image_compat(img: np.ndarray, img_size):
    if mediapy is not None:
        return mediapy.resize_image(img, img_size)
    if cv2 is not None:
        h, w = int(img_size[0]), int(img_size[1])
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if Image is not None:
        h, w = int(img_size[0]), int(img_size[1])
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    return img

def plot_state(
    current_state,
    log_traj = False,
    traj_preds=None, 
    traj_pred_score=None, 
    past_traj_length = 0,
    dx = 75, 
    center_agent_idx = -1, 
    filename = None, 
    t = None, 
    tick_off = False, 
    return_ax = False,
    img_size = (400,400),
    font_size = 12,
    center_xy = None,
    traj_color = 'r',
    is_ego = None,
    is_adv = None,
    show_speed_colorbar = False,
    speed_vmin = 0.0,
    speed_vmax = 20.0,
    traj_linewidth = None,
    traj_alpha = None,
    show_agent_ids = False,
    agent_box_scale = 1.35,
    force_overlap_mask = None,
):
    viz_config = visualization.utils.VizConfig()
    fig, ax = visualization.utils.init_fig_ax(viz_config)
    if log_traj:
        traj = current_state.log_trajectory
    else:
        traj = current_state.sim_trajectory
    indices = np.arange(traj.num_objects) if bool(show_agent_ids) else None
    is_controlled = np.asarray(current_state.object_metadata.is_controlled).astype(bool)
    # INTERACTION-converted scenarios may leave is_controlled all-False.
    # Fallback to current valid mask so overlap highlighting remains visible.
    if (is_controlled.ndim != 1) or (not np.any(is_controlled)):
        is_controlled = np.asarray(traj.valid[:, current_state.timestep]).astype(bool)

    visualization.plot_trajectory(
        ax, traj, is_controlled, time_idx=current_state.timestep, 
        indices=indices, past_traj_length = past_traj_length,
        is_ego = is_ego, is_adv = is_adv,
        show_colorbar=show_speed_colorbar,
        speed_vmin=float(speed_vmin),
        speed_vmax=float(speed_vmax),
        traj_linewidth=traj_linewidth,
        traj_alpha=traj_alpha,
        box_scale=float(agent_box_scale),
        force_overlap_mask=force_overlap_mask,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    visualization.plot_roadgraph_points(ax, current_state.roadgraph_points, verbose=False)
    visualization.plot_traffic_light_signals_as_points(
        ax, current_state.log_traffic_light, current_state.timestep, verbose=False
    )

    current_xy = np.asarray(traj.xy[:, current_state.timestep, :])
    current_valid = np.asarray(traj.valid[:, current_state.timestep])
    if center_xy is not None:
        origin_x, origin_y = center_xy
    elif center_agent_idx == -1:
        is_sdc = np.asarray(current_state.object_metadata.is_sdc).astype(bool)
        sdc_valid_mask = is_sdc & current_valid
        if np.any(sdc_valid_mask):
            xy = current_xy[sdc_valid_mask]
            origin_x, origin_y = xy[0, :2]
        elif np.any(current_valid):
            xy = current_xy[current_valid]
            origin_x, origin_y = xy[0, :2]
        elif current_xy.shape[0] > 0:
            origin_x, origin_y = current_xy[0, :2]
        else:
            origin_x, origin_y = 0.0, 0.0
    else:
        if 0 <= center_agent_idx < current_xy.shape[0]:
            xy = current_xy[center_agent_idx]
            origin_x, origin_y = xy[:2]
        elif np.any(current_valid):
            xy = current_xy[current_valid]
            origin_x, origin_y = xy[0, :2]
        elif current_xy.shape[0] > 0:
            origin_x, origin_y = current_xy[0, :2]
        else:
            origin_x, origin_y = 0.0, 0.0
    # Zoom: support isotropic scalar range or anisotropic (x_half, y_half).
    if isinstance(dx, (tuple, list, np.ndarray)) and len(dx) >= 2:
        x_half = float(dx[0])
        y_half = float(dx[1])
    else:
        x_half = float(dx)
        y_half = float(dx)

    ax.axis((
        origin_x - x_half,
        origin_x + x_half,
        origin_y - y_half,
        origin_y + y_half,
    ))
    if t is None:
        t = (current_state.timestep-10)/10
    if font_size>0:
        ax.text(origin_x - 0.9 * x_half, origin_y + 0.9 * y_half, f"t={t:.1f} s", fontsize=font_size)
    
    if tick_off:
        plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 

    if traj_preds is not None:
        T, D = traj_preds.shape[-2:]
    
        if traj_pred_score is not None:
            
            for traj, score in zip(traj_preds.reshape(-1, T, D), traj_pred_score.reshape(-1)):
                if score < 0.01:
                    continue
                ax.plot(traj[:, 0], traj[:, 1], color=traj_color, alpha=score*0.8+0.2)
        else:
            for traj in traj_preds.reshape(-1, T, D):
                ax.plot(traj[:, 0], traj[:, 1], color=traj_color, alpha=0.8)
        
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0
    )
    if filename is not None:
        plt.savefig(filename,
                    bbox_inches='tight', 
                    transparent=False,
                    pad_inches=0.02)
    if return_ax:
        return fig, ax
    return _resize_image_compat(visualization.utils.img_from_fig(fig), img_size)
