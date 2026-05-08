import sys
import os
import torch
import pickle
import zipfile
import glob
import numpy as np
from torch.utils.data import Dataset
from .data_utils import *
import functools
import pickle


class WaymaxDataset(Dataset):
    """
    Dataset class for Waymax data.
    """

    def __init__(
        self,
        data_dir,
        cache_unidbo_only: bool = True,
    ):
        # Only collect pkl/zip samples; avoid re-reading cached npz files as raw samples.
        if data_dir is not None:
            pkl_list = glob.glob(os.path.join(data_dir, '*.pkl'))
            zip_list = glob.glob(os.path.join(data_dir, '*.zip'))
            self.data_list = sorted(pkl_list + zip_list)
        else:
            self.data_list = []
        self.cache_unidbo_only = cache_unidbo_only
        
        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.data_list)

    def gen_tensor(self, data):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """
        def to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.array(x)

        agents_history = to_numpy(data['agents_history'])
        agents_interested = to_numpy(data['agents_interested'])
        agents_future = to_numpy(data['agents_future'])
        agents_type = to_numpy(data['agents_type'])
        traffic_light_points = to_numpy(data['traffic_light_points'])
        polylines = to_numpy(data['polylines'])
        polylines_valid = to_numpy(data['polylines_valid'])
        relations = to_numpy(data['relations'])

        tensors = {
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations)
        }
        
        return tensors

    def _cache_path(self, pkl_path: str) -> str:
        return f"{pkl_path}.unidbo_only.npz"

    def _extract_unidbo_only(self, data: dict) -> dict:
        # Only keep the keys used by UniDBO training to reduce load overhead.
        needed_keys = [
            'agents_history',
            'agents_future',
            'agents_interested',
            'agents_type',
            'traffic_light_points',
            'polylines',
            'polylines_valid',
            'relations',
            'scenario_id',
        ]
        return {k: data[k] for k in needed_keys if k in data}

    def _load_from_cache_or_pickle(self, path: str) -> dict:
        if self.cache_unidbo_only:
            cache_path = self._cache_path(path)
            if os.path.exists(cache_path):
                # np.load with allow_pickle=False for safety; all arrays are numeric.
                with np.load(cache_path) as npz:
                    return {k: npz[k] for k in npz.files}

        if path.endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zf:
                with zf.open('unidbo.pkl', 'r') as f:
                    data = pickle.load(f)
        else:
            # Deserialize only pkl content to avoid loading cache files incorrectly.
            with open(path, 'rb') as f:
                data = pickle.load(f)

        if self.cache_unidbo_only:
            unidbo_only = self._extract_unidbo_only(data)
            # Save compressed cache to speed up subsequent epochs; best-effort.
            try:
                np.savez_compressed(self._cache_path(path), **unidbo_only)
            except Exception:
                pass
            return unidbo_only

        return data

    def __getitem__(self, idx):
        data = self._load_from_cache_or_pickle(self.data_list[idx])
        return self.gen_tensor(data)


class WaymaxTestDataset(WaymaxDataset):
    """
    Test dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        max_object (int, optional): Maximum number of objects. Defaults to 16.
        max_polylines (int, optional): Maximum number of polylines. Defaults to 256.
        history_length (int, optional): Length of history. Defaults to 11.
        num_points_polyline (int, optional): Number of points in each polyline. Defaults to 30.
    """

    def __init__(
        self,
        data_dir: str,
        max_object: int = 16,
        max_map_points: int = 3000,
        max_polylines: int = 256,
        history_length: int = 11,
        num_points_polyline: int = 30,
    ) -> None:
        super().__init__(data_dir)

        self.max_object = max_object
        self.max_polylines = max_polylines
        self.max_map_points = max_map_points
        self.history_length = history_length
        self.num_points_polyline = num_points_polyline
        
        self.base_path = os.path.dirname(os.path.abspath(self.data_list[0])) if len(self.data_list) > 0 else None
                
    def process_scenario(self, scenario_raw, current_index: int = 10,
                        use_log: bool = True, selected_agents=None,
                        remove_history=False):
        """
        Process a scenario and generate tensors.

        Args:
            scenario_raw (dict): Raw scenario data.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.
            selected_agents (list, optional): List of selected agents. Defaults to None.

        Returns:
            dict: Dictionary of tensors.
        """
        data_dict = data_process_scenario(
            scenario_raw,
            current_index=current_index,
            max_num_objects=self.max_object,
            max_polylines=self.max_polylines,
            num_points_polyline=self.num_points_polyline,
            use_log=use_log,
            selected_agents=selected_agents,
            remove_history=remove_history,
        )
        
        return data_dict
        
    def reset_agent_length(self,max_object):
        """
        Reset the maximum number of objects.

        Args:
            max_object (int): Maximum number of objects.
        """
        self.max_object = max_object
        
    def get_scenario_by_id(
        self, scenario_id,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False
    ):
        """
        Get a scenario by its ID.

        Args:
            scenario_id (int): Scenario ID.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        file_path = os.path.join(self.base_path, f"scenario_{scenario_id}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if 'scenario_raw' in data:
            scenario_raw = data['scenario_raw']
        elif 'scenario' in data:
            scenario_raw = data['scenario']
        else:
            raise ValueError("scenario_raw not found")
        
        data_dict = self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )
        
        return scenario_id, scenario_raw, data_dict
    
    def get_scenario_by_index(
        self, index,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False
    ):
        """
        Get a scenario by its index.

        Args:
            index (int): Scenario index.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        filename = self.data_list[index]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if 'scenario_raw' in data:
            scenario_raw = data['scenario_raw']
            scenario_id = data['scenario_id']
        elif 'scenario' in data:
            scenario_raw = data['scenario']
            scenario_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
        else:
            raise ValueError("scenario_raw not found")
        
        
        data_dict = self.process_scenario(
            scenario_raw,
            current_index=current_index,
            use_log=use_log,
            remove_history=remove_history,
        )
        
        return scenario_id, scenario_raw, data_dict
    
    def __getitem__(self, idx):
        _, _, data_dict = self.get_scenario_by_index(idx)
        return data_dict
