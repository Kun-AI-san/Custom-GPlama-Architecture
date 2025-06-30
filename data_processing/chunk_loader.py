from torch.utils.data import Dataset
import torch

class ChunkLoader(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            output_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(input_chunk)
            self.output_ids.append(output_chunk)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), torch.tensor(self.output_ids[index])