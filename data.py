from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import torch

class AutoRegressiveDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "input_ids": self.dataset[idx]["input_ids"]
        }

def autoregressive_collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    return {
        "input_ids": input_ids  # [B, T]
    }

def get_dataloader(dataset_path, batch_size=8, shuffle=True, num_workers=4):
    dataset = AutoRegressiveDataset(dataset_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=autoregressive_collate_fn
    )

