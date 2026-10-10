"""Checkpoint'i tarayıcıda çalışan üretici için JSON'a aktarır (static/model.json).

Ağırlıklar little-endian float32 olarak base64 ile saklanır; böylece JavaScript
tarafı Python ile birebir aynı sayılarla hesap yapar ve dosya küçük kalır.

Kullanım:
  python -m src.web_export
"""

import base64
from dataclasses import asdict
from pathlib import Path
import struct

import torch

from src.checkpoint import FORMAT_VERSION, Checkpoint
from src.config import CHECKPOINT_PATH, MAX_NAME_LEN, MAX_WORD_LEN, MIN_WORD_LEN, STATIC_DIR
from src.data_loader import Vocabulary
from src.io_utils import write_json

WEB_MODEL_PATH = STATIC_DIR / 'model.json'


def _encode_tensor(tensor: torch.Tensor) -> dict:
  values = tensor.detach().float().flatten().tolist()
  raw = struct.pack(f'<{len(values)}f', *values)
  return {'shape': list(tensor.shape), 'data': base64.b64encode(raw).decode('ascii')}


def build_web_model(checkpoint: Checkpoint) -> dict:
  return {
      'version': FORMAT_VERSION,
      'vocab': checkpoint.vocab.chars,
      'special': {
          'PAD': Vocabulary.PAD,
          'BOS_FIRST': Vocabulary.BOS_FIRST,
          'BOS_LAST': Vocabulary.BOS_LAST,
          'EOS': Vocabulary.EOS,
      },
      'config': asdict(checkpoint.model_config),
      'limits': {
          'min_word_len': MIN_WORD_LEN,
          'max_word_len': MAX_WORD_LEN,
          'max_name_len': MAX_NAME_LEN,
      },
      'known_names': sorted(checkpoint.known_names),
      'tensors': {name: _encode_tensor(t) for name, t in checkpoint.model_state.items()},
  }


def export_web_model(checkpoint_path: Path = CHECKPOINT_PATH, output_path: Path = WEB_MODEL_PATH):
  write_json(output_path, build_web_model(Checkpoint.load(checkpoint_path)), compact=True)
  print(f"Web modeli '{output_path}' konumuna aktarıldı ({output_path.stat().st_size / 1e6:.2f} MB).")


if __name__ == '__main__':
  export_web_model()
