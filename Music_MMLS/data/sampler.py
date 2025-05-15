from random import shuffle
import torch
from torch.utils.data import Sampler

class AccedingSequenceLengthBatchSampler(Sampler):
    def __init__(self, data, batch_size) -> None:
        self.data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sizes = torch.tensor([len(x[1]) for x in self.data])
        for batch in torch.chunk(torch.argsort(sizes), len(self)):
            batch_list = batch.tolist()
            shuffle(batch_list)
            yield batch_list
