"""Dynamic within-batch padding; metadata never enters the classifier."""
import torch

def collate_messages(batch):
    longest = max(item['length'] for item in batch)
    ids = torch.zeros(len(batch), longest, dtype=torch.long)
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        ids[i, :item['length']] = torch.tensor(item['input_ids'])
    return {'input_ids': ids, 'lengths': lengths,
            'targets': torch.tensor([item['target'] for item in batch], dtype=torch.long),
            'message_ids': [item['message_id'] for item in batch],
            'game_ids': [item['game_id'] for item in batch]}
