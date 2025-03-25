"""Portable CPU/GPU inference from Phase 3 state_dict checkpoints."""
from pathlib import Path
import torch
from diplomacy.data.text_dataset import Vocabulary
from diplomacy.models.neural.lstm import StandardLSTM
from diplomacy.training.lstm_trainer import load_checkpoint

class Predictor:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        state = load_checkpoint(Path(checkpoint_path), 'cpu')
        self.vocab = Vocabulary.from_dict(state['vocabulary'])
        self.config = state['experiment_config']
        self.task = state['task']
        self.model = StandardLSTM(vocab_size=len(self.vocab), **state['model_config'])
        self.model.load_state_dict(state['model_state'])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text):
        ids = self.vocab.encode(text, self.config['data']['max_tokens'])
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits = self.model(tensor, torch.tensor([len(ids)], device=self.device))
        probability = float(logits.softmax(dim=1)[0, 1].item())
        return {'task': self.task, 'prediction': int(probability >= 0.5),
                'prob_lie': probability, 'prob_truth': 1 - probability}

def load_predictor(checkpoint_path, device='cpu'):
    return Predictor(checkpoint_path, device=device)
