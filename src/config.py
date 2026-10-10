"""Proje genelindeki dosya yolları, hiperparametreler ve üretim sınırları."""

from dataclasses import dataclass
from pathlib import Path

# Yollar proje köküne göre çözülür; uygulama hangi dizinden başlatılırsa başlatılsın çalışır
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / 'data' / 'raw' / 'nba_players_raw.json'
PROCESSED_DATA_PATH = ROOT_DIR / 'data' / 'processed' / 'clean_names.json'
CHECKPOINT_PATH = ROOT_DIR / 'models' / 'lstm_weights.pth'
STATIC_DIR = ROOT_DIR / 'static'


@dataclass(frozen=True)
class ModelConfig:
  embedding_dim: int = 32
  hidden_dim: int = 128
  num_layers: int = 2
  dropout: float = 0.2


@dataclass(frozen=True)
class TrainConfig:
  epochs: int = 60
  batch_size: int = 64
  lr: float = 3e-3
  val_ratio: float = 0.1
  patience: int = 6
  grad_clip: float = 1.0
  seed: int = 42


# Üretim sınırları: arayüzdeki kart en uzun gerçek isim kadar (24 karakter) metni rahat gösteriyor
MIN_WORD_LEN = 2
MAX_WORD_LEN = 16
MAX_NAME_LEN = 24

DEFAULT_TEMPERATURE = 0.7
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 2.0
