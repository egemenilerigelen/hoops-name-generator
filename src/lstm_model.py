import torch.nn as nn


class CharLSTM(nn.Module):
  """Karakter seviyesinde dil modeli: Embedding -> LSTM -> Linear."""

  def __init__(
      self,
      vocab_size,
      embedding_dim=32,
      hidden_dim=128,
      num_layers=2,
      dropout=0.2,
      padding_idx=0,
  ):
    super().__init__()
    self.embedding = nn.Embedding(
        vocab_size, embedding_dim, padding_idx=padding_idx
    )
    self.lstm = nn.LSTM(
        embedding_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x, hidden=None):
    """x: (batch, seq_len) -> logits: (batch, seq_len, vocab_size).

    Eğitimde tüm dizi tek seferde işlenir. Üretimde hidden state adımlar arasında
    taşınır, böylece model ismin başından itibaren tüm bağlamı hatırlar.
    """
    out, hidden = self.lstm(self.embedding(x), hidden)
    return self.fc(out), hidden
