"""Vocabulary, ad/soyad ayrıştırma ve PyTorch Dataset tanımları."""

from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.io_utils import read_json

FIRST = 'first'
LAST = 'last'

# Kuşak ekleri soyadın parçası değildir: "gary payton ii" -> "payton"
SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
# Soyadın önüne bitişik giden ön ekler: "dick van arsdale" -> "van arsdale"
PARTICLES = {'van', 'vander', 'von', 'de', 'del', 'da', 'di', 'du', 'la', 'le', 'st', 'bon'}


def load_names(path: Path) -> list[str]:
  return read_json(path)


def split_name(full_name: str) -> tuple[str, str] | None:
  """Temizlenmiş tam ismi (ad, soyad) çiftine ayırır; göbek adları ve ekler atılır."""
  parts = full_name.split()
  while len(parts) > 2 and parts[-1] in SUFFIXES:
    parts.pop()
  if len(parts) < 2:
    return None

  last_parts = [parts[-1]]
  if len(parts) > 2 and parts[-2] in PARTICLES:
    last_parts.insert(0, parts[-2])
  return parts[0], ' '.join(last_parts)


def build_samples(names: list[str]) -> list[tuple[str, str]]:
  """Her isimden (tür, kelime) örnekleri çıkarır. Tekrarlar korunur ki frekans öğrenilsin."""
  samples = []
  for name in names:
    split = split_name(name)
    if split is None:
      continue
    first, last = split
    samples.append((FIRST, first))
    samples.append((LAST, last))
  return samples


def known_full_names(names: list[str]) -> set[str]:
  """Üretimde reddedilecek gerçek isimler: ham hali ve ekleri atılmış (ad soyad) hali."""
  known = set(names)
  for name in names:
    split = split_name(name)
    if split is not None:
      known.add(' '.join(split))
  return known


class Vocabulary:
  """Karakter <-> indeks eşlemesi. Özel token'lar sabit indekslerde tutulur.

  Ad ve soyad için ayrı başlangıç token'ları kullanılır; böylece tek model iki
  farklı dağılımı (ör. "Jamal" vs "Washington") koşullu olarak öğrenir.
  """

  PAD = 0
  BOS_FIRST = 1
  BOS_LAST = 2
  EOS = 3
  NUM_SPECIAL = 4

  def __init__(self, chars):
    self.chars = sorted(set(chars))
    self.char2idx = {ch: i + self.NUM_SPECIAL for i, ch in enumerate(self.chars)}
    self.idx2char = {i: ch for ch, i in self.char2idx.items()}

  @classmethod
  def from_words(cls, words) -> 'Vocabulary':
    return cls(''.join(words))

  def __len__(self) -> int:
    return self.NUM_SPECIAL + len(self.chars)

  def bos(self, kind: str) -> int:
    return self.BOS_FIRST if kind == FIRST else self.BOS_LAST

  def encode(self, word: str) -> list[int]:
    return [self.char2idx[ch] for ch in word]

  def decode(self, indices) -> str:
    return ''.join(self.idx2char[i] for i in indices)


class NameDataset(Dataset):
  """Teacher forcing örnekleri: girdi = [BOS] + harfler, hedef = harfler + [EOS]."""

  def __init__(self, samples: list[tuple[str, str]], vocab: Vocabulary):
    self.items = []
    for kind, word in samples:
      encoded = vocab.encode(word)
      self.items.append((
          torch.tensor([vocab.bos(kind)] + encoded),
          torch.tensor(encoded + [vocab.EOS]),
      ))

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    return self.items[idx]


def collate_batch(batch):
  """Farklı uzunluktaki dizileri PAD ile aynı boya getirir."""
  inputs, targets = zip(*batch)
  return (
      pad_sequence(inputs, batch_first=True, padding_value=Vocabulary.PAD),
      pad_sequence(targets, batch_first=True, padding_value=Vocabulary.PAD),
  )
