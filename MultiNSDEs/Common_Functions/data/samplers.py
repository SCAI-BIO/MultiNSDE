from torch.utils.data import Sampler
import numpy as np
import random

class BalancedNonPlaceboSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "The batch_size must be an even number."

        pbo_flags = dataset.get_pbo_flags()
        self.non_pbo_indices = np.where(pbo_flags == 1)[0].tolist()
        self.pbo_indices = np.where(pbo_flags == 0)[0].tolist()

        self.half_batch = batch_size // 2
        self.num_full_batches = len(self.non_pbo_indices) // self.half_batch
        self.remaining = len(self.non_pbo_indices) % self.half_batch

    def __iter__(self):
        no_pbo_shuffled = random.sample(self.non_pbo_indices, len(self.non_pbo_indices))
        total_pbo_needed = self.num_full_batches * self.half_batch + self.remaining
        pbo_sampled = random.choices(self.pbo_indices, k=total_pbo_needed)

        for i in range(self.num_full_batches):
            no_pbo_batch = no_pbo_shuffled[i*self.half_batch:(i+1)*self.half_batch]
            pbo_batch = pbo_sampled[i*self.half_batch:(i+1)*self.half_batch]
            batch = no_pbo_batch + pbo_batch
            random.shuffle(batch)
            yield batch

        if self.remaining > 0:
            no_pbo_last = no_pbo_shuffled[-self.remaining:]
            pbo_last = pbo_sampled[-self.remaining:]
            batch = no_pbo_last + pbo_last
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_full_batches + (1 if self.remaining > 0 else 0)
