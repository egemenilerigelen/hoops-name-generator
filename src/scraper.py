import json
import os
from nba_api.stats.static import players

def fetch_all_nba_players(output_path: str = "data/raw/nba_players_raw.json"):
    """
    nba_api kullanarak NBA tarihinde oynamış tüm oyuncuların isimlerini çeker
    ve ham JSON dosyası olarak kaydeder.
    """
    print("NBA resmi veritabanına bağlanılıyor...")
    
    # Tüm zamanların oyuncu listesini getir (Aktif + Emekli)
    all_players = players.get_players()
    
    # Sadece tam isimleri (full_name) filtrele
    player_names = [player['full_name'] for player in all_players]
    
    # Çıktı klasörünün varlığını kontrol et, yoksa oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # JSON olarak diske kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(player_names, f, ensure_ascii=False, indent=4)
        
    print(f"İşlem Tamamlandı! Toplam {len(player_names)} oyuncu ismi '{output_path}' konumuna kaydedildi.")

if __name__ == "__main__":
    fetch_all_nba_players()