from src.lstm_model import LSTMPipeline

def main():
    print("Hoops Name Generator'a Hoş Geldiniz!")
    print("-" * 40)
    
    # LSTM Pipeline'ı başlat
    pipeline = LSTMPipeline(data_path="data/processed/clean_names.json")
    
    # Modeli eğit (Bunu sadece ilk çalıştırmada veya parametre değiştirince yapman yeterli)
    # Eğer model ağırlıklarını kaydetme mekanizması eklersen bu adımı her seferinde yapmana gerek kalmaz.
    pipeline.train_model(epochs=15, batch_size=128)
    
    print("\nÜretilen LSTM Tabanlı Basketbolcu İsimleri:")
    print("=" * 40)
    for i in range(1, 11):
        # Temperature parametresini 0.6 ile 0.9 arası bir değer verebilirsin
        isim = pipeline.generate_full_name(temperature=0.7)
        print(f"{i}. {isim}")

if __name__ == "__main__":
    main()