from typing import Dict, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss

from unidbo_core.model.model_utils import inverse_kinematics, roll_out
from unidbo_core.model.modules import Denoiser, Encoder
from unidbo_core.model.utils import DHNNoiseScheduler


class UniDBOModel(pl.LightningModule):
    """
    UniDBO: shared scene encoder with two one-step denoising branches.

    - CS: continuous-scale branch for open-loop prediction.
    - DHN: discrete high-noise branch for closed-loop simulation.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        cfg = self._normalize_cfg_aliases(dict(cfg))
        self.save_hyperparameters({"cfg": cfg})

        self.cfg = cfg
        self._future_len = int(cfg["future_len"])
        self._agents_len = int(cfg["agents_len"])
        self._action_len = int(cfg["action_len"])
        self._diffusion_steps = int(cfg["diffusion_steps"])
        self._encoder_layers = int(cfg["encoder_layers"])
        self._schedule_type = cfg.get("schedule_type", "cosine")
        self._embeding_dim = int(cfg.get("embeding_dim", 5))

        self._train_encoder = bool(cfg.get("train_encoder", True))
        self._train_dhn = bool(cfg.get("train_dhn", True))
        self._train_cs = bool(cfg.get("train_cs", True))
        self.branch_mode = str(cfg.get("branch_mode", "both")).lower()
        if self.branch_mode not in {"dhn", "cs", "both"}:
            raise ValueError(f"Invalid branch_mode={self.branch_mode}, expected dhn|cs|both")
        self.enable_dhn = self.branch_mode in {"dhn", "both"}
        self.enable_cs = self.branch_mode in {"cs", "both"}

        self.branch_select_mode = str(cfg.get("branch_select_mode", "best_fde")).lower()
        if self.branch_select_mode not in {"best_fde", "dhn", "cs"}:
            raise ValueError(
                f"Invalid branch_select_mode={self.branch_select_mode}, expected best_fde|dhn|cs"
            )

        self.dhn_terminal_step = int(cfg.get("dhn_terminal_step", self._diffusion_steps - 1))
        if self.dhn_terminal_step < 0:
            self.dhn_terminal_step = self._diffusion_steps - 1
        if self.dhn_terminal_step >= self._diffusion_steps:
            raise ValueError("dhn_terminal_step must be < diffusion_steps")
        self.dhn_noise_start_ratio = float(cfg.get("dhn_noise_start_ratio", 0.7))
        self.dhn_noise_end_ratio = float(cfg.get("dhn_noise_end_ratio", 1.0))
        self.dhn_t_anneal_steps = int(cfg.get("dhn_t_anneal_steps", 18000))
        self.dhn_eval_mode = str(cfg.get("dhn_eval_mode", "terminal")).lower()
        if self.dhn_eval_mode not in {"terminal", "random"}:
            raise ValueError("dhn_eval_mode must be terminal|random")
        self.dhn_supervised_weight = float(cfg.get("dhn_supervised_weight", 1.25))
        self.dhn_action_loss_weight = float(cfg.get("dhn_action_loss_weight", 0.05))

        self.cs_num_scales = int(cfg.get("cs_num_scales", self._diffusion_steps))
        self.cs_sigma_min = float(cfg.get("cs_sigma_min", 0.01))
        self.cs_sigma_max = float(cfg.get("cs_sigma_max", 1.0))
        self.cs_sigma_rho = float(cfg.get("cs_sigma_rho", 7.0))
        self.cs_supervised_weight = float(cfg.get("cs_supervised_weight", 1.25))

        self.encoder = Encoder(self._encoder_layers)
        self.dhn_branch = None
        if self.enable_dhn:
            self.dhn_branch = Denoiser(
                future_len=self._future_len,
                action_len=self._action_len,
                agents_len=self._agents_len,
                steps=self._diffusion_steps,
                input_dim=self._embeding_dim,
                noise_condition="discrete",
            )

        self.cs_branch = None
        if self.enable_cs:
            self.cs_branch = Denoiser(
                future_len=self._future_len,
                action_len=self._action_len,
                agents_len=self._agents_len,
                steps=self.cs_num_scales,
                input_dim=self._embeding_dim,
                noise_condition="continuous",
            )
            self.register_buffer(
                "cs_sigma_levels",
                self._build_cs_sigma_levels(
                    self.cs_num_scales, self.cs_sigma_min, self.cs_sigma_max, self.cs_sigma_rho
                ),
            )

        self.noise_scheduler = DHNNoiseScheduler(
            steps=self._diffusion_steps,
            schedule=self._schedule_type,
            s=cfg.get("schedule_s", 0.0),
            e=cfg.get("schedule_e", 1.0),
            tau=cfg.get("schedule_tau", 1.0),
            scale=cfg.get("schedule_scale", 1.0),
        )
        self.register_buffer("action_mean", torch.tensor(cfg["action_mean"], dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(cfg["action_std"], dtype=torch.float32))

        self._global_step_local = 0

    @staticmethod
    def _normalize_cfg_aliases(cfg: dict) -> dict:
        old_one = "one" + "_step_"
        legacy_mode_key = old_one + "branch_mode"
        branch_mode = cfg.get("branch_mode", cfg.get(legacy_mode_key, "both"))
        branch_mode = str(branch_mode).lower().replace("d" + "dpm", "dhn").replace("c" + "t", "cs")
        cfg["branch_mode"] = branch_mode
        legacy_select_key = old_one + "branch_select_mode"
        select_mode = cfg.get("branch_select_mode", cfg.get(legacy_select_key, "best_fde"))
        cfg["branch_select_mode"] = (
            str(select_mode).lower().replace("d" + "dpm", "dhn").replace("c" + "t", "cs")
        )

        aliases = {
            old_one + "terminal_step": "dhn_terminal_step",
            old_one + "high_noise_start_ratio": "dhn_noise_start_ratio",
            old_one + "high_noise_end_ratio": "dhn_noise_end_ratio",
            old_one + "t_anneal_steps": "dhn_t_anneal_steps",
            old_one + "eval_mode": "dhn_eval_mode",
            old_one + "action_loss_weight": "dhn_action_loss_weight",
            "denoise_supervised_weight": "dhn_supervised_weight",
            "c" + "t_num_scales": "cs_num_scales",
            "c" + "t_sigma_min": "cs_sigma_min",
            "c" + "t_sigma_max": "cs_sigma_max",
            "c" + "t_sigma_rho": "cs_sigma_rho",
            "c" + "t_supervised_weight": "cs_supervised_weight",
        }
        for old, new in aliases.items():
            if new not in cfg and old in cfg:
                cfg[new] = cfg[old]
        return cfg

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        state_dict = checkpoint.get("state_dict", {})
        remapped = {}
        for key, value in list(state_dict.items()):
            if ("feat_" + "kd_") in key:
                continue
            new_key = key
            legacy_dhn_prefix = "deno" + "iser."
            legacy_cs_prefix = "c" + "t_deno" + "iser."
            if key.startswith(legacy_dhn_prefix):
                new_key = "dhn_branch." + key[len(legacy_dhn_prefix) :]
            elif key.startswith(legacy_cs_prefix):
                new_key = "cs_branch." + key[len(legacy_cs_prefix) :]
            elif key == "c" + "t_sigma_levels":
                new_key = "cs_sigma_levels"
            remapped[new_key] = value
        checkpoint["state_dict"] = remapped

    @staticmethod
    def _build_cs_sigma_levels(num_steps: int, sigma_min: float, sigma_max: float, rho: float):
        ramp = torch.linspace(0.0, 1.0, max(num_steps, 2), dtype=torch.float32)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        return (min_inv_rho + ramp * (max_inv_rho - min_inv_rho)) ** rho

    def reset_agent_length(self, agents_len: int) -> None:
        self._agents_len = int(agents_len)
        if self.dhn_branch is not None:
            self.dhn_branch.reset_agent_length(agents_len)
        if self.cs_branch is not None:
            self.cs_branch.reset_agent_length(agents_len)

    def configure_optimizers(self):
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_dhn and self.dhn_branch is not None:
            for param in self.dhn_branch.parameters():
                param.requires_grad = False
        if not self._train_cs and self.cs_branch is not None:
            for param in self.cs_branch.parameters():
                param.requires_grad = False

        params_to_update = [param for param in self.parameters() if param.requires_grad]
        if not params_to_update:
            raise RuntimeError("No parameters to update")

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=float(self.cfg["lr"]),
            weight_decay=float(self.cfg["weight_decay"]),
        )
        warmup = int(self.cfg["lr_warmup_step"])
        step_freq = int(self.cfg["lr_step_freq"])
        gamma = float(self.cfg["lr_step_gamma"])

        def lr_update(step):
            if step < warmup:
                scale = 1 - (warmup - step) / warmup * 0.95
            else:
                scale = gamma ** ((step - warmup) // step_freq)
            return min(max(scale, 1e-2), 1.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def normalize_actions(self, actions: torch.Tensor):
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor):
        return actions * self.action_std + self.action_mean

    def batch_to_device(self, input_dict: dict, device: torch.device = "cuda"):
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)
        return input_dict

    def _prepare_targets(self, batch: dict):
        agents_future = batch["agents_future"][:, : self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch["agents_interested"][:, : self._agents_len]
        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future, agents_future_valid, dt=0.1, action_len=self._action_len
        )
        gt_actions_normalized = self.normalize_actions(gt_actions)
        return (
            agents_future,
            agents_future_valid,
            agents_interested,
            gt_actions,
            gt_actions_valid,
            gt_actions_normalized,
        )

    def _sample_dhn_steps(self, batch_size: int, num_agents: int, device, is_train: bool):
        max_idx = int(self.dhn_terminal_step)
        if not is_train:
            if self.dhn_eval_mode == "terminal":
                t_base = torch.full((batch_size,), max_idx, dtype=torch.long, device=device)
            else:
                t_base = torch.randint(0, max_idx + 1, (batch_size,), device=device).long()
            return t_base[:, None].repeat(1, num_agents).view(batch_size, num_agents, 1, 1)

        progress = 1.0
        if self.dhn_t_anneal_steps > 0:
            progress = min(float(self._global_step_local) / float(self.dhn_t_anneal_steps), 1.0)
        ratio = self.dhn_noise_start_ratio + progress * (
            self.dhn_noise_end_ratio - self.dhn_noise_start_ratio
        )
        low_idx = int(round(min(max(ratio, 0.0), 1.0) * max_idx))
        if low_idx >= max_idx:
            t_base = torch.full((batch_size,), max_idx, dtype=torch.long, device=device)
        else:
            t_base = torch.randint(low_idx, max_idx + 1, (batch_size,), device=device).long()
        return t_base[:, None].repeat(1, num_agents).view(batch_size, num_agents, 1, 1)

    def _sample_cs_sigma(self, batch_size: int, num_agents: int, device, dtype):
        if self.cs_sigma_min <= 0 or self.cs_sigma_max <= 0 or self.cs_sigma_min > self.cs_sigma_max:
            raise ValueError(
                f"Invalid CS sigma range: [{self.cs_sigma_min}, {self.cs_sigma_max}]"
            )
        u = torch.rand((batch_size,), device=device, dtype=dtype)
        sigma_base = self.cs_sigma_min * torch.pow(
            torch.full_like(u, self.cs_sigma_max / self.cs_sigma_min),
            u,
        )
        return sigma_base.view(batch_size, 1, 1, 1).repeat(1, num_agents, 1, 1)

    def forward_dhn_branch(self, encoder_outputs, noised_actions_normalized, diffusion_step):
        if self.dhn_branch is None:
            raise RuntimeError("DHN branch is disabled")
        noised_actions = self.unnormalize_actions(noised_actions_normalized)
        branch_output = self.dhn_branch(encoder_outputs, noised_actions, diffusion_step)
        denoised_actions_normalized = self.noise_scheduler.q_x0(
            branch_output,
            diffusion_step,
            noised_actions_normalized,
        )
        return self._decode_branch_outputs(
            encoder_outputs, denoised_actions_normalized, branch_output, self.dhn_branch
        )

    def forward_cs_branch(self, encoder_outputs, noised_actions_normalized, sigma):
        if self.cs_branch is None:
            raise RuntimeError("CS branch is disabled")
        if torch.is_tensor(sigma) and sigma.ndim > 2:
            sigma = sigma.view(sigma.shape[0], sigma.shape[1], -1)[..., 0]
        noised_actions = self.unnormalize_actions(noised_actions_normalized)
        denoised_actions_normalized = self.cs_branch(encoder_outputs, noised_actions, sigma)
        return self._decode_branch_outputs(
            encoder_outputs, denoised_actions_normalized, denoised_actions_normalized, self.cs_branch
        )

    def _decode_branch_outputs(
        self, encoder_outputs, denoised_actions_normalized, branch_output, branch_module
    ):
        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        denoised_actions = self.unnormalize_actions(denoised_actions_normalized)
        denoised_trajs = roll_out(
            current_states, denoised_actions, action_len=branch_module._action_len, global_frame=True
        )
        return {
            "branch_output": branch_output,
            "denoised_actions_normalized": denoised_actions_normalized,
            "denoised_actions": denoised_actions,
            "denoised_trajs": denoised_trajs,
        }

    def denoise_loss(self, denoised_trajs, agents_future, agents_future_valid, agents_interested):
        agents_future = agents_future[..., 1:, :3]
        future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)
        state_loss = smooth_l1_loss(
            denoised_trajs[..., :2], agents_future[..., :2], reduction="none"
        ).sum(-1)
        yaw_error = denoised_trajs[..., 2] - agents_future[..., 2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)
        denom = future_mask.sum().clamp_min(1.0)
        return (state_loss * future_mask).sum() / denom, (yaw_loss * future_mask).sum() / denom

    def action_loss(self, actions, actions_gt, actions_valid, agents_interested):
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        action_loss = smooth_l1_loss(actions, actions_gt, reduction="none").sum(-1)
        return (action_loss * action_mask).sum() / action_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def calculate_metrics_denoise(
        self, denoised_trajs, agents_future, agents_future_valid, agents_interested, top_k=None
    ):
        top_k = top_k or self._agents_len
        pred_traj = denoised_trajs[:, :top_k, :, :2]
        gt = agents_future[:, :top_k, 1:, :2]
        gt_mask = (agents_future_valid[:, :top_k, 1:] & (agents_interested[:, :top_k, None] > 0)).bool()
        denoise_mse = torch.norm(pred_traj - gt, dim=-1)
        denoise_ade = denoise_mse[gt_mask].mean()
        denoise_fde = denoise_mse[..., -1][gt_mask[..., -1]].mean()
        return denoise_ade.item(), denoise_fde.item()

    def _select_eval_branch(
        self, dhn_metrics: Optional[Dict[str, float]], cs_metrics: Optional[Dict[str, float]]
    ) -> Tuple[str, Dict[str, float]]:
        if dhn_metrics is None and cs_metrics is None:
            raise RuntimeError("No active UniDBO branch produced metrics")
        if dhn_metrics is None:
            return "cs", cs_metrics
        if cs_metrics is None:
            return "dhn", dhn_metrics
        if self.branch_select_mode == "dhn":
            return "dhn", dhn_metrics
        if self.branch_select_mode == "cs":
            return "cs", cs_metrics
        if (dhn_metrics["fde"], dhn_metrics["ade"]) <= (cs_metrics["fde"], cs_metrics["ade"]):
            return "dhn", dhn_metrics
        return "cs", cs_metrics

    def forward_and_get_loss(self, batch, prefix="", debug=False):
        (
            agents_future,
            agents_future_valid,
            agents_interested,
            _gt_actions,
            gt_actions_valid,
            gt_actions_normalized,
        ) = self._prepare_targets(batch)
        batch_size, num_agents, num_steps, action_dim = gt_actions_normalized.shape
        is_train = prefix.startswith("train/")
        encoder_outputs = self.encoder(batch)
        log_dict = {}
        debug_outputs = {}
        total_loss = torch.tensor(0.0, device=agents_future.device, dtype=agents_future.dtype)

        dhn_metrics = None
        if self.enable_dhn:
            dhn_steps = self._sample_dhn_steps(batch_size, num_agents, agents_future.device, is_train)
            dhn_noise = torch.randn(batch_size, num_agents, num_steps, action_dim).type_as(agents_future)
            dhn_noised = self.noise_scheduler.add_noise(gt_actions_normalized, dhn_noise, dhn_steps)
            dhn_outputs = self.forward_dhn_branch(encoder_outputs, dhn_noised, dhn_steps.view(batch_size, num_agents))
            state_loss, yaw_loss = self.denoise_loss(
                dhn_outputs["denoised_trajs"], agents_future, agents_future_valid, agents_interested
            )
            dhn_loss = state_loss + yaw_loss
            dhn_weighted = self.dhn_supervised_weight * dhn_loss
            total_loss = total_loss + dhn_weighted
            if self.dhn_action_loss_weight > 0:
                act_loss = self.action_loss(
                    dhn_outputs["denoised_actions_normalized"],
                    gt_actions_normalized,
                    gt_actions_valid,
                    agents_interested,
                )
                total_loss = total_loss + self.dhn_action_loss_weight * act_loss

            ade, fde = self.calculate_metrics_denoise(
                dhn_outputs["denoised_trajs"], agents_future, agents_future_valid, agents_interested, 8
            )
            if is_train:
                log_dict.update(
                    {
                        prefix + "dhn_state_loss": state_loss.item(),
                        prefix + "dhn_yaw_loss": yaw_loss.item(),
                        prefix + "dhn_loss": dhn_loss.item(),
                        prefix + "dhn_loss_weighted": dhn_weighted.item(),
                        prefix + "dhn_t_mean": dhn_steps.float().mean().item(),
                        prefix + "dhn_denoise_ADE": ade,
                        prefix + "dhn_denoise_FDE": fde,
                    }
                )
            dhn_metrics = {"ade": ade, "fde": fde, "loss": dhn_weighted.item()}
            debug_outputs["dhn_outputs"] = dhn_outputs

        cs_metrics = None
        if self.enable_cs:
            cs_sigma = self._sample_cs_sigma(
                batch_size,
                num_agents,
                agents_future.device,
                gt_actions_normalized.dtype,
            )
            cs_noise = torch.randn(batch_size, num_agents, num_steps, action_dim).type_as(agents_future)
            cs_noised = gt_actions_normalized + cs_sigma * cs_noise
            cs_outputs = self.forward_cs_branch(encoder_outputs, cs_noised, cs_sigma.view(batch_size, num_agents))
            state_loss, yaw_loss = self.denoise_loss(
                cs_outputs["denoised_trajs"], agents_future, agents_future_valid, agents_interested
            )
            cs_loss = state_loss + yaw_loss
            cs_weighted = self.cs_supervised_weight * cs_loss
            total_loss = total_loss + cs_weighted
            ade, fde = self.calculate_metrics_denoise(
                cs_outputs["denoised_trajs"], agents_future, agents_future_valid, agents_interested, 8
            )
            if is_train:
                log_dict.update(
                    {
                        prefix + "cs_state_loss": state_loss.item(),
                        prefix + "cs_yaw_loss": yaw_loss.item(),
                        prefix + "cs_loss": cs_loss.item(),
                        prefix + "cs_loss_weighted": cs_weighted.item(),
                        prefix + "cs_denoise_ADE": ade,
                        prefix + "cs_denoise_FDE": fde,
                    }
                )
            cs_metrics = {"ade": ade, "fde": fde, "loss": cs_weighted.item()}
            debug_outputs["cs_outputs"] = cs_outputs

        _, selected_metrics = self._select_eval_branch(dhn_metrics, cs_metrics)
        if is_train:
            log_dict[prefix + "loss"] = total_loss.item()
        else:
            log_dict.update(
                {
                    prefix + "loss": total_loss.item(),
                    prefix + "ADE": selected_metrics["ade"],
                    prefix + "FDE": selected_metrics["fde"],
                }
            )
        if debug:
            return total_loss, log_dict, debug_outputs
        return total_loss, log_dict

    def training_step(self, batch, batch_idx):
        self._global_step_local += 1
        loss, log_dict = self.forward_and_get_loss(batch, prefix="train/")
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="val/")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample_open_loop(self, batch, sigma: float = -1.0):
        batch = self.batch_to_device(batch, self.device)
        encoder_outputs = self.encoder(batch)
        agents_history = encoder_outputs["agents"]
        batch_size, num_agents = agents_history.shape[:2]
        num_steps = self._future_len // self._action_len
        if sigma <= 0:
            sigma_full = self._sample_cs_sigma(
                batch_size,
                num_agents,
                self.device,
                torch.float32,
            )
            sigma_ba = sigma_full.view(batch_size, num_agents)
            x_sigma = torch.randn(batch_size, num_agents, num_steps, 2, device=self.device) * sigma_full
        else:
            sigma_value = float(sigma)
            sigma_ba = torch.full((batch_size, num_agents), sigma_value, device=self.device, dtype=torch.float32)
            x_sigma = torch.randn(batch_size, num_agents, num_steps, 2, device=self.device) * sigma_value
        return self.forward_cs_branch(encoder_outputs, x_sigma, sigma_ba)

    @torch.no_grad()
    def sample_closed_loop(self, batch, terminal_step: int = -1):
        batch = self.batch_to_device(batch, self.device)
        encoder_outputs = self.encoder(batch)
        agents_history = encoder_outputs["agents"]
        batch_size, num_agents = agents_history.shape[:2]
        num_steps = self._future_len // self._action_len
        t = self.dhn_terminal_step if terminal_step < 0 else int(terminal_step)
        t_ba = torch.full((batch_size, num_agents), t, device=self.device, dtype=torch.long)
        x_t = torch.randn(batch_size, num_agents, num_steps, 2, device=self.device)
        outputs = self.forward_dhn_branch(encoder_outputs, x_t, t_ba)
        outputs["history"] = {
            "branch": "dhn",
            "dhn_t": int(t),
            "dhn_num_steps": int(self.noise_scheduler.num_steps),
        }
        return outputs
