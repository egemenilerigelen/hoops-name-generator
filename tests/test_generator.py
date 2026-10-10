import pytest
import torch

from src.config import MAX_NAME_LEN
from src.data_loader import Vocabulary
from src.generator import VALID_WORD, NameGenerationError, NameGeneratorService
from src.lstm_model import CharLSTM


@pytest.fixture(scope='module')
def service():
  return NameGeneratorService.from_checkpoint()


@pytest.mark.parametrize('temperature', [0.2, 0.7, 1.2])
def test_generated_names_are_novel_and_fit_ui(service, temperature):
  torch.manual_seed(0)
  for _ in range(30):
    name = service.generate(temperature)
    first, last = name.lower().split(' ', 1)
    assert name.lower() not in service.known_names
    assert len(name) <= MAX_NAME_LEN
    assert VALID_WORD.match(first) and VALID_WORD.match(last)
    assert name == name.title()


@pytest.mark.parametrize('temperature', [0.0, -1.0, 5.0])
def test_rejects_out_of_range_temperature(service, temperature):
  with pytest.raises(ValueError):
    service.generate(temperature)


def test_raises_when_no_valid_name_possible():
  # Hiç EOS üretmeyen bir model kelime sınırını aşar ve geçerli isim üretemez
  vocab = Vocabulary('ab')
  model = CharLSTM(len(vocab))
  torch.nn.init.zeros_(model.fc.weight)
  with torch.no_grad():
    model.fc.bias.fill_(-1e4)
    model.fc.bias[vocab.char2idx['a']] = 1e4
  service = NameGeneratorService(model, vocab, known_names=set())
  with pytest.raises(NameGenerationError):
    service.generate(0.7, max_attempts=3)
