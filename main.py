from src.lstm_model import LSTMPipeline

def main():
    print("Hoops Name Generator'a Hoş Geldiniz!")
    print("-" * 40)
    
    # LSTM Pipeline'ı başlat
    pipeline = LSTMPipeline(data_path="data/processed/clean_names.json")
    
    # Eğitimi tekrar çalıştırmak yerine dondurulmuş ağırlıkları yüklüyoruz
    weights_path = "models/lstm_weights.pth"
    if not pipeline.load_weights(weights_path):
        print("Kayıtlı ağırlık bulunamadı, model eğitiliyor...")
        pipeline.train_model(epochs=15, batch_size=128)
        pipeline.save_weights(weights_path)
    
    print("\nÜretilen LSTM Tabanlı Basketbolcu İsimleri:")
    print("=" * 40)
    for i in range(1, 11):
        isim = pipeline.generate_full_name(temperature=0.7)
        print(f"{i}. {isim}")

if __name__ == "__main__":
    main()