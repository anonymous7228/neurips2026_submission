from typing import Optional, Sequence

import jax
import numpy as np
import torch
from jax import numpy as jnp
from waymax import datatypes
from waymax.agents import actor_core


def sample_to_action(
    sample: np.ndarray,
    is_controlled: jax.Array,
    agents_id: Optional[Sequence[int]] = None,
    max_num_objects: int = 128,
) -> actor_core.WaymaxActorOutput:
    """Convert UniDBO action samples to a Waymax actor output."""
    action_dim = sample.shape[-1]
    actions_array = np.zeros((max_num_objects, action_dim), dtype=sample.dtype)

    control_mask = np.asarray(is_controlled).astype(bool)
    if agents_id is None:
        agent_indices = np.arange(control_mask.shape[0])
    elif len(agents_id) == control_mask.shape[0]:
        agent_indices = np.asarray(agents_id)
    elif len(agents_id) < control_mask.shape[0]:
        agent_indices = np.full(control_mask.shape[0], -1, dtype=int)
        agent_indices[control_mask] = np.asarray(agents_id)
    else:
        raise ValueError("Invalid agents_id size")

    for sample_idx, agent_idx in enumerate(agent_indices):
        if agent_idx >= 0:
            actions_array[agent_idx] = sample[sample_idx]

    actions_valid = np.zeros((max_num_objects, 1), dtype=bool)
    valid_indices = agent_indices[agent_indices >= 0]
    actions_valid[valid_indices] = True

    actions = datatypes.Action(
        data=jnp.asarray(actions_array),
        valid=jnp.asarray(actions_valid),
    )
    return actor_core.WaymaxActorOutput(
        action=actions,
        actor_state=None,
        is_controlled=jnp.asarray(actions_valid).squeeze(-1),
    )


def duplicate_batch(batch: dict, num_samples: int):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            assert value.shape[0] == 1, "Only support batch size of 1"
            batch[key] = torch.cat([value] * num_samples, dim=0)
    return batch


def torch_dict_to_numpy(input_dict: dict):
    output = {}
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.detach().cpu().numpy()
        else:
            output[key] = value
    return output


def stack_dict(items: list):
    if not items:
        return {}

    output = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if isinstance(values[0], np.ndarray):
            output[key] = np.stack(values, axis=0)
        elif isinstance(values[0], dict):
            output[key] = stack_dict(values)
        else:
            output[key] = values
    return output
