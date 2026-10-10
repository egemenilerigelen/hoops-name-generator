import math

import torch

from src.checkpoint import Checkpoint, create_model
from src.config import ModelConfig, TrainConfig
from src.data_loader import NameDataset, Vocabulary, build_samples
from src.generator import NameGeneratorService
from src.trainer import Trainer, build_loaders

NAMES = ['michael jordan', 'magic johnson', 'larry bird', 'kobe bryant', 'tim duncan', 'kevin garnett']
SMALL_MODEL = ModelConfig(embedding_dim=8, hidden_dim=16, num_layers=1, dropout=0.0)
QUICK_TRAINING = TrainConfig(epochs=2, batch_size=4, val_ratio=0.25, patience=5)


def _setup():
  samples = build_samples(NAMES)
  vocab = Vocabulary.from_words(word for _, word in samples)
  train_loader, val_loader = build_loaders(NameDataset(samples, vocab), QUICK_TRAINING)
  return vocab, train_loader, val_loader


def test_trainer_returns_best_state():
  torch.manual_seed(0)
  vocab, train_loader, val_loader = _setup()
  model = create_model(vocab, SMALL_MODEL)

  result = Trainer(model, train_loader, val_loader, QUICK_TRAINING, torch.device('cpu')).fit()

  assert result.epochs_run == QUICK_TRAINING.epochs
  assert math.isfinite(result.best_val_loss)
  assert result.best_state.keys() == model.state_dict().keys()


def test_checkpoint_roundtrip_feeds_generator(tmp_path):
  vocab, _, _ = _setup()
  model = create_model(vocab, SMALL_MODEL)
  path = tmp_path / 'model.pth'

  Checkpoint(model.state_dict(), vocab, SMALL_MODEL, frozenset(NAMES)).save(path)
  loaded = Checkpoint.load(path)

  assert loaded.vocab.chars == vocab.chars
  assert loaded.model_config == SMALL_MODEL
  assert loaded.known_names == frozenset(NAMES)
  for name, tensor in loaded.build_model().state_dict().items():
    assert torch.equal(tensor, model.state_dict()[name])
  assert isinstance(NameGeneratorService.from_checkpoint(path), NameGeneratorService)
