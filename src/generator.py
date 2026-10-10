"""Eğitilmiş checkpoint'ten özgün isim üreten servis katmanı."""

from pathlib import Path
import re

import torch

from src.config import CHECKPOINT_PATH, DEFAULT_TEMPERATURE, MAX_NAME_LEN, MAX_TEMPERATURE, MAX_WORD_LEN, MIN_TEMPERATURE, MIN_WORD_LEN
from src.data_loader import FIRST, LAST, Vocabulary
from src.lstm_model import CharLSTM

# Boşluk/tire yalnızca iki harf grubunun arasında olabilir: "van exel", "dejean-jones"
VALID_WORD = re.compile(r'^[a-z]+(?:[ -][a-z]+)*$')


class NameGenerationError(RuntimeError):
  """Deneme sınırı içinde geçerli ve özgün bir isim üretilemediğinde fırlatılır."""


class NameGeneratorService:

  def __init__(self, model: CharLSTM, vocab: Vocabulary, known_names: set[str], device='cpu'):
    self.model = model.to(device).eval()
    self.vocab = vocab
    self.known_names = known_names
    self.device = torch.device(device)

  @classmethod
  def from_checkpoint(cls, path: Path = CHECKPOINT_PATH, device='cpu') -> 'NameGeneratorService':
    # weights_only=True: checkpoint içinde keyfi Python nesnesi çalıştırılmasını engeller
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    vocab = Vocabulary(checkpoint['vocab_chars'])
    model = CharLSTM(len(vocab), padding_idx=Vocabulary.PAD, **checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state'])
    return cls(model, vocab, set(checkpoint['known_names']), device)

  @torch.inference_mode()
  def generate_word(self, kind: str, temperature: float) -> str | None:
    """Tek bir ad ya da soyadı karakter karakter örnekler. Geçersizse None döner."""
    token = torch.tensor([[self.vocab.bos(kind)]], device=self.device)
    hidden = None
    indices = []

    for _ in range(MAX_WORD_LEN + 1):
      logits, hidden = self.model(token, hidden)
      logits = logits[0, -1] / temperature
      # Özel token'lar kelime içinde asla örneklenmemeli
      logits[[Vocabulary.PAD, Vocabulary.BOS_FIRST, Vocabulary.BOS_LAST]] = -torch.inf
      if len(indices) < MIN_WORD_LEN:
        logits[Vocabulary.EOS] = -torch.inf

      next_idx = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
      if next_idx == Vocabulary.EOS:
        word = self.vocab.decode(indices)
        return word if VALID_WORD.match(word) else None

      indices.append(next_idx)
      token = torch.tensor([[next_idx]], device=self.device)

    return None

  def generate(self, temperature: float = DEFAULT_TEMPERATURE, max_attempts: int = 50) -> str:
    """Eğitim verisinde birebir bulunmayan, arayüze sığan bir tam isim üretir."""
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
      raise ValueError(f'temperature {MIN_TEMPERATURE} ile {MAX_TEMPERATURE} arasında olmalı')

    for _ in range(max_attempts):
      first = self.generate_word(FIRST, temperature)
      last = self.generate_word(LAST, temperature)
      if first is None or last is None:
        continue

      full_name = f'{first} {last}'
      if len(full_name) <= MAX_NAME_LEN and full_name not in self.known_names:
        return full_name.title()

    raise NameGenerationError(f'{max_attempts} denemede özgün bir isim üretilemedi')


if __name__ == '__main__':
  service = NameGeneratorService.from_checkpoint()
  for temp in (0.3, 0.7, 1.1):
    print(f'\nTemperature {temp}:')
    for i in range(1, 6):
      print(f'  {i}. {service.generate(temp)}')
