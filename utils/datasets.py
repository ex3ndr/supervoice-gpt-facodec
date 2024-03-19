import torch
import numpy as np
import random
import os
from pathlib import Path

def create_dataset_loader(path, input_length, output_length, batch_size, tokenizer, workers):

    # Load dataset
    files = []
    with open(path, 'r') as file:
        for line in file:
            files.append(line.strip())
    dataset = PreprocessedDataset(dir = str(Path(path).parent), files = files, tokenizer = tokenizer, max_input_length = input_length, max_output_length = output_length + 1)

    # Collator
    def collate_fn(batch):
        B = len(batch)
        x, y = zip(*batch)

        # Calculate lengths
        x_lengths = torch.tensor([len(x) for x in x])
        y_lengths = torch.tensor([len(y) - 1 for y in y])

        # Calculate sizes
        input_length_t = max([len(x) for x in x])
        output_length_t = max([len(y) for y in y]) - 1

        # # Create targets
        t = torch.zeros((B, output_length_t, 4), dtype = torch.int64)
        for i in range(B):
            t[i, :len(y[i]) - 1] = y[i][1:]

        # Padded tensors
        x_padded = torch.IntTensor(B, input_length_t)
        y_padded = torch.IntTensor(B, output_length_t, 4)
        x_padded.zero_()
        y_padded.zero_()
        for i in range(B):
            x_padded[i, :len(x[i])] = x[i]
            y_padded[i, :y[i].shape[0] - 1,:] = y[i][:-1,:]

        return x_padded, y_padded, t, x_lengths, y_lengths

    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = workers, shuffle=False, pin_memory=True, drop_last=True, collate_fn = collate_fn)
    return loader

#
# Dataset Classes
# 

class PreprocessedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir, files, tokenizer, max_input_length, max_output_length):
        self.files = files
        self.dir = dir
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def generate(self):
        while True:

            # Pick random id
            id = random.choice(self.files)

            # Load text
            with open(os.path.join(self.dir, f"{id}.txt"), 'r') as file:
                text = file.read()
            text_tokens = self.tokenizer.encode(text) if random.random() < 0.3 else self.tokenizer.encode_sample(text) # 30% chance of using optimal tokenization
            text_tokens = torch.tensor(text_tokens)
            text_tensor = torch.cat([torch.tensor([self.tokenizer.sequence_begin_token_id]),  text_tokens, torch.tensor([self.tokenizer.sequence_end_token_id])]).int()

            # Load codes
            codes = torch.load(os.path.join(self.dir, f"{id}.codec.pt"), map_location = "cpu")
            codes = codes[0:4] # Only use first 4 codes
            codes = codes.transpose(0, 1) # Transpose to [time, codes]
            codes_tensor = torch.cat([torch.tensor([[self.tokenizer.sequence_begin_token_id, 0, 0, 0]]), codes + 3, torch.tensor([[self.tokenizer.sequence_end_token_id, 0, 0, 0]])]).int()

            # Check if sequence is too long: sample again
            if len(text_tensor) > self.max_input_length or len(codes_tensor) > self.max_output_length:
                continue

            yield (text_tensor, codes)

    def __iter__(self):
        return iter(self.generate())