import json
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

from src.config import MAX_NAME_LEN
from src.data_loader import Vocabulary
from src.generator import NameGeneratorService
from src.web_export import WEB_MODEL_PATH, build_web_model

JS_RUNNER = Path(__file__).parent / 'js_runner.cjs'
requires_node = pytest.mark.skipif(shutil.which('node') is None, reason='Node.js kurulu değil')


def run_js(*args) -> list:
  result = subprocess.run(
      ['node', str(JS_RUNNER), str(WEB_MODEL_PATH), *map(str, args)],
      capture_output=True, text=True, encoding='utf-8', check=True,
  )
  return json.loads(result.stdout)


def test_web_model_matches_checkpoint():
  # Model yeniden eğitilip `python -m src.web_export` unutulursa bu test yakalar
  with open(WEB_MODEL_PATH, encoding='utf-8') as f:
    assert json.load(f) == build_web_model()


@requires_node
def test_js_forward_pass_matches_pytorch():
  service = NameGeneratorService.from_checkpoint()
  tokens = [Vocabulary.BOS_LAST] + service.vocab.encode('van exel')

  with torch.inference_mode():
    expected, _ = service.model(torch.tensor([tokens]))

  actual = torch.tensor(run_js('logits', ','.join(map(str, tokens))))
  torch.testing.assert_close(actual, expected[0], atol=1e-4, rtol=1e-4)


@requires_node
def test_js_generates_novel_names():
  service = NameGeneratorService.from_checkpoint()
  names = run_js('generate', 0.7, 30)
  assert len(names) == 30
  for name in names:
    assert name == name.title()
    assert len(name) <= MAX_NAME_LEN
    assert name.lower() not in service.known_names
