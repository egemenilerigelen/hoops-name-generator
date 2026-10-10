"""Eğitim döngüsü (Trainer) ve uçtan uca eğitim akışı (run_training)."""

from dataclasses import dataclass
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.checkpoint import Checkpoint, create_model
from src.config import CHECKPOINT_PATH, PROCESSED_DATA_PATH, ModelConfig, TrainConfig
from src.data_loader import NameDataset, Vocabulary, build_samples, collate_batch, known_full_names, load_names


@dataclass(frozen=True)
class TrainingResult:
  best_val_loss: float
  best_state: dict[str, torch.Tensor]
  epochs_run: int


class Trainer:
  """Verilen modeli verilen veri yükleyicilerle eğitir.

  Veri hazırlama ve diske kaydetme bu sınıfın sorumluluğu değildir; bu sayede
  farklı model ya da veriyle (ör. testlerde küçük örneklerle) yeniden kullanılabilir.
  """

  def __init__(
      self,
      model: nn.Module,
      train_loader: DataLoader,
      val_loader: DataLoader,
      config: TrainConfig,
      device: torch.device,
      ignore_index: int = Vocabulary.PAD,
  ):
    self.model = model.to(device)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.config = config
    self.device = device
    self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

  def fit(self) -> TrainingResult:
    """Erken durdurma ile eğitir ve en iyi doğrulama kaybına ait ağırlıkları döndürür."""
    best_val_loss, best_state, stale_epochs = math.inf, None, 0
    epoch = 0

    for epoch in range(1, self.config.epochs + 1):
      train_loss = self._run_epoch(self.train_loader, training=True)
      val_loss = self._run_epoch(self.val_loader, training=False)
      print(f'Epoch [{epoch}/{self.config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

      if val_loss < best_val_loss:
        best_val_loss, stale_epochs = val_loss, 0
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
      else:
        stale_epochs += 1
        if stale_epochs >= self.config.patience:
          print(f'Doğrulama kaybı {self.config.patience} epoch boyunca iyileşmedi, eğitim erken durduruldu.')
          break

    return TrainingResult(best_val_loss, best_state, epoch)

  def _run_epoch(self, loader: DataLoader, training: bool) -> float:
    self.model.train(training)
    total_loss, total_items = 0.0, 0

    with torch.set_grad_enabled(training):
      for inputs, targets in loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits, _ = self.model(inputs)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if training:
          self.optimizer.zero_grad()
          loss.backward()
          nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
          self.optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_items += inputs.size(0)

    return total_loss / total_items


def build_loaders(dataset: Dataset, config: TrainConfig) -> tuple[DataLoader, DataLoader]:
  """Veriyi tekrarlanabilir şekilde eğitim/doğrulama olarak böler."""
  val_size = int(len(dataset) * config.val_ratio)
  train_set, val_set = random_split(
      dataset,
      [len(dataset) - val_size, val_size],
      generator=torch.Generator().manual_seed(config.seed),
  )
  return (
      DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch),
      DataLoader(val_set, batch_size=config.batch_size, collate_fn=collate_batch),
  )


def run_training(
    data_path: Path = PROCESSED_DATA_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_config: ModelConfig = ModelConfig(),
    train_config: TrainConfig = TrainConfig(),
) -> TrainingResult:
  """Veriyi hazırlar, modeli eğitir ve en iyi ağırlıkları checkpoint olarak kaydeder."""
  random.seed(train_config.seed)
  torch.manual_seed(train_config.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  names = load_names(data_path)
  samples = build_samples(names)
  vocab = Vocabulary.from_words(word for _, word in samples)
  dataset = NameDataset(samples, vocab)
  train_loader, val_loader = build_loaders(dataset, train_config)
  print(f'{len(names)} isim, {len(dataset)} örnek ({len(train_loader.dataset)} eğitim / {len(val_loader.dataset)} doğrulama), cihaz: {device}')

  model = create_model(vocab, model_config)
  result = Trainer(model, train_loader, val_loader, train_config, device).fit()

  Checkpoint(result.best_state, vocab, model_config, frozenset(known_full_names(names))).save(checkpoint_path)
  print(f"En iyi model (Val Loss: {result.best_val_loss:.4f}) '{checkpoint_path}' konumuna kaydedildi.")
  return result
