"""Model eğitimi ve kendi kendine yeten checkpoint kaydı."""

from dataclasses import asdict
import math
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config import CHECKPOINT_PATH, PROCESSED_DATA_PATH, ModelConfig, TrainConfig
from src.data_loader import NameDataset, Vocabulary, build_samples, collate_batch, known_full_names, load_names
from src.lstm_model import CharLSTM

CHECKPOINT_VERSION = 2


def _run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=None):
  training = optimizer is not None
  model.train(training)
  total_loss, total_items = 0.0, 0

  with torch.set_grad_enabled(training):
    for inputs, targets in loader:
      inputs, targets = inputs.to(device), targets.to(device)
      logits, _ = model(inputs)
      loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

      if training:
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

      total_loss += loss.item() * inputs.size(0)
      total_items += inputs.size(0)

  return total_loss / total_items


def save_checkpoint(path: Path, model_state, vocab: Vocabulary, model_config: ModelConfig, known_names):
  """Ağırlıklar, vocabulary, mimari ve bilinen isimler tek dosyada saklanır.

  Böylece çıkarım (inference) veri dosyalarına bağımlı kalmaz ve veri değişse
  bile kayıtlı ağırlıklarla indeksler kaymaz.
  """
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
      {
          'version': CHECKPOINT_VERSION,
          'model_state': model_state,
          'vocab_chars': vocab.chars,
          'model_config': asdict(model_config),
          'known_names': sorted(known_names),
      },
      path,
  )


def train(
    data_path: Path = PROCESSED_DATA_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_config: ModelConfig = ModelConfig(),
    train_config: TrainConfig = TrainConfig(),
) -> float:
  """Modeli eğitir, en iyi doğrulama kaybına sahip ağırlıkları kaydeder ve o kaybı döndürür."""
  random.seed(train_config.seed)
  torch.manual_seed(train_config.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  names = load_names(data_path)
  samples = build_samples(names)
  vocab = Vocabulary.from_words(word for _, word in samples)
  dataset = NameDataset(samples, vocab)

  val_size = int(len(dataset) * train_config.val_ratio)
  train_set, val_set = random_split(
      dataset,
      [len(dataset) - val_size, val_size],
      generator=torch.Generator().manual_seed(train_config.seed),
  )
  train_loader = DataLoader(train_set, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_batch)
  val_loader = DataLoader(val_set, batch_size=train_config.batch_size, collate_fn=collate_batch)

  model = CharLSTM(len(vocab), padding_idx=Vocabulary.PAD, **asdict(model_config)).to(device)
  criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
  optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

  print(f'{len(names)} isim, {len(dataset)} örnek ({len(train_set)} eğitim / {val_size} doğrulama), cihaz: {device}')

  best_val_loss, best_state, stale_epochs = math.inf, None, 0
  for epoch in range(1, train_config.epochs + 1):
    train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, train_config.grad_clip)
    val_loss = _run_epoch(model, val_loader, criterion, device)
    print(f'Epoch [{epoch}/{train_config.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
      best_val_loss, stale_epochs = val_loss, 0
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
      stale_epochs += 1
      if stale_epochs >= train_config.patience:
        print(f'Doğrulama kaybı {train_config.patience} epoch boyunca iyileşmedi, eğitim erken durduruldu.')
        break

  save_checkpoint(checkpoint_path, best_state, vocab, model_config, known_full_names(names))
  print(f"En iyi model (Val Loss: {best_val_loss:.4f}) '{checkpoint_path}' konumuna kaydedildi.")
  return best_val_loss
