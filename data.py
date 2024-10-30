from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
import torch
import gymnasium as gym
import dsrl

class SafeGymnasium(Dataset):
    def __init__(self,env_name,split='train',subtraj_len=100,traj_len=1000,**kwargs):
        env = gym.make('OfflineCarGoal2-v0')
        data = env.get_dataset()
        action_dim = data['actions'].shape[1]
        ob_dim = data['observations'].shape[1]
        self.observations = data['observations']
        self.next_observations = data['next_observations']
        self.actions = data['actions']
        num_traj = self.observations.shape[0]
        num_train = int(num_traj*0.9)
        if split == 'train':
            self.num_traj = num_train
            self.offset = 0
        else:
            self.num_traj = num_traj - num_train
            self.offset = num_train
        
    def __len__(self):
        return self.num_traj
    
    def __getitem__(self, idx):
        state = self.observations[self.offset+idx]
        next_state = self.next_observations[self.offset+idx]
        action = self.actions[self.offset+idx]
        return torch.tensor(state), torch.tensor(next_state), torch.tensor(action)

# class SafeGymnasium(Dataset):
#     def __init__(self,env_name,split='train',subtraj_len=100,traj_len=1000,**kwargs):
#         env = gym.make('OfflineCarGoal2-v0')
#         data = env.get_dataset()
#         action_dim = data['actions'].shape[1]
#         ob_dim = data['observations'].shape[1]
#         self.observations = data['observations'].reshape(-1, subtraj_len, ob_dim)
#         self.next_observations = data['next_observations'].reshape(-1, subtraj_len, ob_dim)
#         self.actions = data['actions'].reshape(-1, subtraj_len, action_dim)
#         num_traj = self.observations.shape[0]
#         num_train = int(num_traj*0.9)
#         if split == 'train':
#             self.num_traj = num_train
#             self.offset = 0
#         else:
#             self.num_traj = num_traj - num_train
#             self.offset = num_train
        
#     def __len__(self):
#         return self.num_traj
    
#     def __getitem__(self, idx):
#         state = self.observations[self.offset+idx]
#         next_state = self.next_observations[self.offset+idx]
#         action = self.actions[self.offset+idx]
#         return torch.tensor(state), torch.tensor(next_state), torch.tensor(action)

class SafeGymnasiumDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        env_name,
        subtraj_len: int = 100,
        train_batch_size: int = 16,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.subtraj_len = subtraj_len
        self.env_name = env_name

    def setup(self, stage=None):

        self.train_dataset = SafeGymnasium(
            env_name = self.env_name,
            subtraj_len = self.subtraj_len,
            split='train',
        )
        
        self.val_dataset = SafeGymnasium(
            env_name = self.env_name,
            subtraj_len = self.subtraj_len,
            split='val',
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )