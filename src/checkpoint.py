"""Eğitilmiş modelin disk formatı.

Format yalnızca burada tanımlanır; eğitim, üretim ve web dışa aktarımı bu sınıf
üzerinden çalışır ve sözlük anahtarlarını bilmek zorunda kalmaz.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.config import ModelConfig
from src.data_loader import Vocabulary
from src.lstm_model import CharLSTM

FORMAT_VERSION = 2


def create_model(vocab: Vocabulary, config: ModelConfig) -> CharLSTM:
  return CharLSTM(len(vocab), padding_idx=Vocabulary.PAD, **asdict(config))


@dataclass(frozen=True)
class Checkpoint:
  """Ağırlıklar + vocabulary + mimari + bilinen isimler.

  Çıkarım veri dosyalarına bağımlı kalmaz; veri değişse bile vocabulary
  indeksleri ağırlıklarla uyumlu kalır.
  """

  model_state: dict[str, torch.Tensor]
  vocab: Vocabulary
  model_config: ModelConfig
  known_names: frozenset[str]

  def save(self, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'version': FORMAT_VERSION,
            'model_state': self.model_state,
            'vocab_chars': self.vocab.chars,
            'model_config': asdict(self.model_config),
            'known_names': sorted(self.known_names),
        },
        path,
    )

  @classmethod
  def load(cls, path: Path) -> 'Checkpoint':
    # weights_only=True: dosya içinde keyfi Python nesnesi çalıştırılmasını engeller
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('version') != FORMAT_VERSION:
      raise ValueError(f"Desteklenmeyen checkpoint sürümü: {data.get('version')} ('{path}')")
    return cls(
        model_state=data['model_state'],
        vocab=Vocabulary(data['vocab_chars']),
        model_config=ModelConfig(**data['model_config']),
        known_names=frozenset(data['known_names']),
    )

  def build_model(self) -> CharLSTM:
    model = create_model(self.vocab, self.model_config)
    model.load_state_dict(self.model_state)
    return model.eval()
