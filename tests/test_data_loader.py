import pytest
import torch

from src.data_loader import FIRST, LAST, NameDataset, Vocabulary, build_samples, collate_batch, known_full_names, split_name
from src.preprocessing import NamePipeline


@pytest.mark.parametrize(
    'raw, expected',
    [
        ('Luka Dončić', 'luka doncic'),
        ('Nikola Jokić', 'nikola jokic'),
        ("Shaquille O'Neal", 'shaquille oneal'),
        ('  Dorian   Finney-Smith ', 'dorian finney-smith'),
    ],
)
def test_clean_name(raw, expected):
  assert NamePipeline().clean_name(raw) == expected


@pytest.mark.parametrize(
    'full_name, expected',
    [
        ('michael jordan', ('michael', 'jordan')),
        ('gary payton ii', ('gary', 'payton')),
        ('jaren jackson jr', ('jaren', 'jackson')),
        ('dick van arsdale', ('dick', 'van arsdale')),
        ('charles r jones', ('charles', 'jones')),
        ('bryce dejean-jones', ('bryce', 'dejean-jones')),
        ('nene', None),
    ],
)
def test_split_name(full_name, expected):
  assert split_name(full_name) == expected


def test_known_full_names_includes_suffix_free_form():
  known = known_full_names(['gary payton ii'])
  assert {'gary payton ii', 'gary payton'} <= known


def test_vocabulary_roundtrip_and_reserved_indices():
  vocab = Vocabulary.from_words(['van exel', 'dejean-jones'])
  encoded = vocab.encode('van-exel')
  assert vocab.decode(encoded) == 'van-exel'
  assert min(encoded) >= Vocabulary.NUM_SPECIAL
  assert len(vocab) == Vocabulary.NUM_SPECIAL + len(set('van exeldejean-jones'))


def test_dataset_shifts_targets_and_pads():
  samples = build_samples(['al horford'])
  assert samples == [(FIRST, 'al'), (LAST, 'horford')]

  vocab = Vocabulary.from_words(word for _, word in samples)
  inputs, targets = collate_batch(list(NameDataset(samples, vocab)))

  assert inputs.shape == targets.shape == (2, len('horford') + 1)
  assert inputs[0, 0] == Vocabulary.BOS_FIRST and inputs[1, 0] == Vocabulary.BOS_LAST
  # Hedef dizi, girdinin bir adım kaydırılmış hali + EOS
  assert torch.equal(targets[1, :-1], inputs[1, 1:])
  assert targets[1, -1] == Vocabulary.EOS
  # Kısa örnek PAD ile doldurulur
  assert targets[0, 2] == Vocabulary.EOS and targets[0, 3:].eq(Vocabulary.PAD).all()
