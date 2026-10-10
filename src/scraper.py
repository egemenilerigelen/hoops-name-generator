"""NBA tarihinde oynamış tüm oyuncuların isimlerini nba_api ile çeker."""

from nba_api.stats.static import players


def fetch_player_names() -> list[str]:
  """Aktif ve emekli tüm oyuncuların tam isimlerini döndürür."""
  return [player['full_name'] for player in players.get_players()]
