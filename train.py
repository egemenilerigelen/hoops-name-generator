"""Uçtan uca eğitim hattı: (isteğe bağlı) veri çekme -> temizleme -> eğitim -> web modeli.

Kullanım:
  python train.py            # mevcut temiz veriden modeli yeniden eğitir
  python train.py --scrape   # önce NBA oyuncu listesini yeniden çeker
"""

import argparse

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH
from src.io_utils import read_json, write_json
from src.preprocessing import NameCleaner
from src.trainer import run_training
from src.web_export import export_web_model


def main():
  parser = argparse.ArgumentParser(description='Hoops Name Generator eğitim hattı')
  parser.add_argument('--scrape', action='store_true', help='NBA verisini nba_api ile yeniden çek')
  args = parser.parse_args()

  if args.scrape:
    # nba_api yalnızca veri toplarken gerekir, bu yüzden geç import ediliyor
    from src.scraper import fetch_player_names
    raw_names = fetch_player_names()
    write_json(RAW_DATA_PATH, raw_names)
    print(f"{len(raw_names)} oyuncu ismi '{RAW_DATA_PATH}' konumuna kaydedildi.")

  if args.scrape or not PROCESSED_DATA_PATH.exists():
    raw_names = read_json(RAW_DATA_PATH)
    clean_names = NameCleaner().clean_all(raw_names)
    write_json(PROCESSED_DATA_PATH, clean_names)
    print(f'{len(raw_names)} ham isimden {len(clean_names)} benzersiz temiz isim elde edildi.')

  run_training()
  # Tarayıcıdaki üretici (GitHub Pages) yeni ağırlıkları kullansın
  export_web_model()


if __name__ == '__main__':
  main()
