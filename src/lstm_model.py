import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class NameDataset(Dataset):

  def __init__(self, names, seq_len=4):
    self.seq_len = seq_len
    chars = set(''.join(names))
    chars.add('^')
    chars.add('$')
    self.chars = sorted(list(chars))

    self.vocab_size = len(self.chars)
    self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
    self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    self.data = []
    for name in names:
      padded = ('^' * seq_len) + name + '$'
      for i in range(len(padded) - seq_len):
        x_seq = padded[i : i + seq_len]
        y_target = padded[i + seq_len]
        self.data.append((x_seq, y_target))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    x_seq, y_target = self.data[idx]
    x_idx = [self.char2idx[ch] for ch in x_seq]
    y_idx = self.char2idx[y_target]
    return torch.tensor(x_idx), torch.tensor(y_idx)


class CharLSTM(nn.Module):

  def __init__(
      self, vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1
  ):
    super(CharLSTM, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(
        embedding_dim, hidden_dim, num_layers, batch_first=True
    )
    self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x, hidden):
    embeds = self.embedding(x)
    out, hidden = self.lstm(embeds, hidden)
    out = self.fc(out[:, -1, :])
    return out, hidden

  def init_hidden(self, batch_size, device):
    return (
        torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
        torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
    )


class LSTMPipeline:

  def __init__(self, data_path='data/processed/clean_names.json'):
    with open(data_path, 'r', encoding='utf-8') as f:
      self.raw_names = json.load(f)

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    words = []
    for full_name in self.raw_names:
      words.extend(full_name.split())

    self.dataset = NameDataset(words, seq_len=4)
    self.model = CharLSTM(
        vocab_size=self.dataset.vocab_size,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
    ).to(self.device)

  def train_model(self, epochs=15, batch_size=128, lr=0.005):
    dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    print('Model eğitimi başlıyor...')
    self.model.train()

    for epoch in range(epochs):
      total_loss = 0
      for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
        optimizer.zero_grad()
        hidden = self.model.init_hidden(x_batch.size(0), self.device)
        outputs, _ = self.model(x_batch, hidden)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

      avg_loss = total_loss / len(dataloader)
      print(f'Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}')

  def save_weights(self, model_path='models/lstm_weights.pth'):
    """Modelin öğrendiği tüm ağırlıkları diske kaydeder."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(self.model.state_dict(), model_path)
    print(f"Model ağırlıkları '{model_path}' konumuna kaydedildi.")

  def load_weights(self, model_path='models/lstm_weights.pth'):
    """Diskteki ağırlıkları modele yükler (Eğitimi atlar)."""
    if os.path.exists(model_path):
      self.model.load_state_dict(
          torch.load(model_path, map_location=self.device)
      )
      self.model.eval()
      print(f"Ağırlıklar '{model_path}' dosyasından başarıyla yüklendi.")
      return True
    return False

  def generate_word(self, temperature=0.8, max_len=15):
    self.model.eval()
    current_seq = '^' * self.dataset.seq_len
    generated_chars = []

    with torch.no_grad():
      for _ in range(max_len):
        x_idx = [self.dataset.char2idx[ch] for ch in current_seq]
        x_tensor = torch.tensor([x_idx]).to(self.device)
        hidden = self.model.init_hidden(1, self.device)
        output, _ = self.model(x_tensor, hidden)

        output = output.squeeze() / temperature
        probs = torch.softmax(output, dim=0)

        next_idx = torch.multinomial(probs, 1).item()
        next_char = self.dataset.idx2char[next_idx]

        if next_char == '$':
          break

        generated_chars.append(next_char)
        current_seq = current_seq[1:] + next_char

    return ''.join(generated_chars).title()

  def generate_full_name(self, temperature=0.8):
    first = self.generate_word(temperature)
    last = self.generate_word(temperature)
    return f'{first} {last}'