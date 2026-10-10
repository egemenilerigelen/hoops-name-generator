"""Ham oyuncu isimlerini modelin kullanacağı temiz Latin formatına dönüştürür."""

import re
import unicodedata


class NameCleaner:
  """Saf metin dönüşümleri yapar; dosya okuma/yazma ile ilgilenmez."""

  # Unicode normalizasyonunun tek başına yakalayamadığı özel harf dönüşümleri
  CUSTOM_MAPPING = {
      'ı': 'i',
      'İ': 'i',
      'đ': 'd',
      'Đ': 'd',
      'ø': 'o',
      'Ø': 'o',
      'æ': 'ae',
      'Æ': 'ae',
      'ß': 'ss',
  }
  MIN_LENGTH = 3

  def to_latin(self, text: str) -> str:
    """Aksanlı ve özel karakterleri standart Latin karşılıklarına dönüştürür.

    Örn: "Dončić" -> "Doncic", "Jokić" -> "Jokic"
    """
    for char, replacement in self.CUSTOM_MAPPING.items():
      text = text.replace(char, replacement)

    # NFKD aksanları harften ayırır, ardından aksan işaretleri atılır
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))

  def clean(self, raw_name: str) -> str:
    """Tek bir ismi küçük harf, yalnızca a-z / boşluk / tire içerecek şekilde temizler."""
    name = self.to_latin(raw_name).lower().strip()
    name = re.sub(r'[^a-z\s-]', '', name)
    return re.sub(r'\s+', ' ', name)

  def clean_all(self, raw_names: list[str]) -> list[str]:
    """Tüm isimleri temizler, çok kısa olanları ve tekrarları atar, sıralı döndürür."""
    cleaned = {self.clean(name) for name in raw_names}
    return sorted(name for name in cleaned if len(name) >= self.MIN_LENGTH)
