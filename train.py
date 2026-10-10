"""Uçtan uca eğitim hattı: (isteğe bağlı) veri çekme -> temizleme -> eğitim.

Kullanım:
  python train.py            # mevcut temiz veriden modeli yeniden eğitir
  python train.py --scrape   # önce NBA oyuncu listesini yeniden çeker
"""

import argparse

from src.config import PROCESSED_DATA_PATH
from src.preprocessing import NamePipeline
from src.trainer import train


def main():
  parser = argparse.ArgumentParser(description='Hoops Name Generator eğitim hattı')
  parser.add_argument('--scrape', action='store_true', help='NBA verisini nba_api ile yeniden çek')
  args = parser.parse_args()

  if args.scrape:
    # nba_api yalnızca veri toplarken gerekir, bu yüzden geç import ediliyor
    from src.scraper import fetch_all_nba_players
    fetch_all_nba_players()

  if args.scrape or not PROCESSED_DATA_PATH.exists():
    NamePipeline().process_and_save()

  train()


if __name__ == '__main__':
  main()
