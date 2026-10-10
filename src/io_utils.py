"""JSON dosya okuma/yazma yardımcıları; dosya erişimi iş mantığından ayrı tutulur."""

import json
from pathlib import Path


def read_json(path: Path):
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f)


def write_json(path: Path, data, *, compact: bool = False):
  """Klasörü gerekirse oluşturur. compact=True boşluksuz yazar (web'e gönderilen dosyalar için)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'w', encoding='utf-8') as f:
    if compact:
      json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    else:
      json.dump(data, f, ensure_ascii=False, indent=4)
