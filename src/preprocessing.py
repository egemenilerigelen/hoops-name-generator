import json
from pathlib import Path
import re
import unicodedata

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH


class NamePipeline:

  def __init__(self):
    # Unicode normalizasyonunun tek başına yakalayamadığı özel harf dönüşüm haritası
    self.custom_mapping = {
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

  def to_latin(self, text: str) -> str:
    """Aksanlı ve özel karakterleri standart Latin (a-z) karşılıklarına dönüştürür.

    Örn: "Dončić" -> "Doncic", "Jokić" -> "Jokic"
    """
    for char, replacement in self.custom_mapping.items():
      text = text.replace(char, replacement)

    # NFKD Normalizasyonu (Aksanları harften ayırır)
    nfkd_form = unicodedata.normalize('NFKD', text)

    # Aksan işaretlerini filtrele
    latin_text = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

    return latin_text

  def clean_name(self, raw_name: str) -> str:
    """İsmi tamamen temizler ve model için ideal formata getirir."""
    name = self.to_latin(raw_name)
    name = name.lower().strip()

    # Sadece küçük latin harfleri, boşluk ve tireyi tut
    name = re.sub(r'[^a-z\s-]', '', name)

    # Birden fazla yan yana boşluğu teke indir
    name = re.sub(r'\s+', ' ', name)

    return name

  def process_and_save(
      self,
      input_path: Path = RAW_DATA_PATH,
      output_path: Path = PROCESSED_DATA_PATH,
  ):
    """Raw JSON verisini okur, pipeline'dan geçirir ve işlenmiş veriyi kaydeder."""
    if not input_path.exists():
      print(
          f"[Hata] '{input_path}' dosyası bulunamadı! Önce"
          ' `python train.py --scrape` çalıştırın.'
      )
      return

    print('Ham veriler işleniyor ve temizleniyor...')

    with open(input_path, 'r', encoding='utf-8') as f:
      raw_names = json.load(f)

    cleaned_names_set = set()
    for raw_name in raw_names:
      cleaned = self.clean_name(raw_name)
      # En az 3 karakterli mantıklı isimleri al
      if len(cleaned) >= 3:
        cleaned_names_set.add(cleaned)

    cleaned_names_list = sorted(list(cleaned_names_set))

    # Çıktı klasörünü kontrol et
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
      json.dump(cleaned_names_list, f, ensure_ascii=False, indent=4)

    print(
        f'İşlem Tamamlandı! {len(raw_names)} ham isimden {len(cleaned_names_list)} adet benzersiz ve temizlenmiş isim elde edildi.'
    )
    print(f"İşlenmiş veri '{output_path}' konumuna kaydedildi.")


if __name__ == '__main__':
  pipeline = NamePipeline()
  pipeline.process_and_save()