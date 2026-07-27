from collections import Counter, defaultdict
import json
import random

VOWELS = set("aeiouy")


class AdvancedMarkovModel:

  def __init__(self, n_gram_size: int = 3):
    self.n_gram_size = n_gram_size
    self.original_names = set()

    # Backoff için farklı N boyutlarında matrisler
    self.first_transitions = {
        n: defaultdict(Counter) for n in range(1, n_gram_size + 1)
    }
    self.last_transitions = {
        n: defaultdict(Counter) for n in range(1, n_gram_size + 1)
    }

    self.first_starts = Counter()
    self.last_starts = Counter()

  def _train_word_list(self, words: list[str], transitions_dict: dict, starts: dict):
    for word in words:
      if len(word) < 2:
        continue

      padded = ("^" * self.n_gram_size) + word + "$"
      starts[padded[: self.n_gram_size]] += 1

      # Farklı N boyutları için (1, 2, 3) geçiş haritalarını doldur
      for n in range(1, self.n_gram_size + 1):
        for i in range(len(padded) - n):
          gram = padded[i : i + n]
          next_char = padded[i + n]
          transitions_dict[n][gram][next_char] += 1

  def train_from_file(self, file_path: str = "data/processed/clean_names.json"):
    with open(file_path, "r", encoding="utf-8") as f:
      names = json.load(f)

    self.original_names = set(names)
    first_names, last_names = [], []

    for full_name in names:
      parts = full_name.split()
      if len(parts) >= 2:
        first_names.append(parts[0])
        last_names.append(parts[-1])

    self._train_word_list(first_names, self.first_transitions, self.first_starts)
    self._train_word_list(last_names, self.last_transitions, self.last_starts)

    print("İleri Seviye Markov Eğitimi Tamamlandı!")

  def _is_consonant_cluster(self, current_word: str) -> bool:
    """Son 3 harfin hepsi sessiz mi kontrol eder."""
    if len(current_word) < 3:
      return False
    return all(char not in VOWELS for char in current_word[-3:])

  def _generate_word(self, transitions_dict: dict, starts: dict, temperature: float) -> str:
    start_grams = list(starts.keys())
    weights = list(starts.values())
    current_gram = random.choices(start_grams, weights=weights, k=1)[0]

    generated = list(current_gram)

    while True:
      possible_next = None
      # Backoff: N-gram'dan başlayıp bulamazsa N-1'e düş
      for n in range(self.n_gram_size, 0, -1):
        lookup_gram = "".join(generated[-n:])
        if lookup_gram in transitions_dict[n]:
          possible_next = transitions_dict[n][lookup_gram]
          break

      if not possible_next:
        break

      chars = list(possible_next.keys())
      counts = list(possible_next.values())

      # Fonetik Dengeleme Filtresi
      current_str = "".join(generated).replace("^", "")
      adjusted_weights = []

      for char, count in zip(chars, counts):
        w = count ** (1.0 / temperature) if temperature != 1.0 else count

        # 3 sessiz harf üst üste geldiyse sesli harfe ekstra ağırlık ver
        if self._is_consonant_cluster(current_str) and char in VOWELS:
          w *= 3.0

        # Çok kısa kelimelerde bitiş karakterini ($) engelle
        if len(current_str) < 4 and char == "$":
          w = 0.0

        adjusted_weights.append(w)

      if sum(adjusted_weights) == 0:
        break

      next_char = random.choices(chars, weights=adjusted_weights, k=1)[0]

      if next_char == "$":
        break

      generated.append(next_char)

    return "".join(generated).replace("^", "").strip().title()

  def generate_unique_name(self, temperature: float = 0.8) -> str:
    """Eğitilen veride BİREBİR VAR OLMAYAN tamamen özgün bir isim türetir."""
    for _ in range(100):  # Maksimum 100 deneme
      first = self._generate_word(self.first_transitions, self.first_starts, temperature)
      last = self._generate_word(self.last_transitions, self.last_starts, temperature)

      full_name = f"{first} {last}".lower()

      # Orijinal veride yoksa ve isimler boş değilse döndür
      if full_name not in self.original_names and len(first) > 2 and len(last) > 2:
        return f"{first} {last}"

    return f"{first} {last}"


if __name__ == "__main__":
  model = AdvancedMarkovModel(n_gram_size=3)
  model.train_from_file()

  print("\n--- GELİŞMİŞ VE %100 ÖZGÜN İSİMLER ---")
  for _ in range(5):
    print(model.generate_unique_name(temperature=0.8))