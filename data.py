from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
import torch
import gymnasium as gym
import dsrl

class SafeGymnasium(Dataset):
    def __init__(self,env_name,split='train',traj_len=1000,**kwargs):
        env = gym.make('OfflineCarGoal2-v0')
        self.data = env.get_dataset()
        self.traj_len = traj_len
        num_traj = self.data['observation'].shape[0]/self.tranj_len
        num_train = int(num_traj*0.9)
        if split == 'train':
            self.num_traj = num_train
            self.offset = 0
        else:
            self.num_traj = num_traj - num_train
            self.offset = num_train
        
    def __len__(self):
        return self.data['observation'].shape[0]/self.tranj_len
    
    def __getitem__(self, idx):
        idx_head = self.offset+1000*idx
        idx_tail = self.offset+1000*(idx+1)
        state = self.data['observation'][idx_head:idx_tail]
        next_state = self.data['next_observations'][idx_head:idx_tail]
        action = self.data['action'][idx_head:idx_tail]
        return torch.tensor(state), torch.tensor(next_state), torch.tensor(action)

class DeepAccidentDataset(LightningDataModule):
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
        self.env_name = env_name

    def setup(self, stage=None):

        self.train_dataset = SafeGymnasium(
            env_name = self.env_name,
            split='train',
        )
        
        self.val_dataset = SafeGymnasium(
            env_name = self.env_name,
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