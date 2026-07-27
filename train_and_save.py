from src.lstm_model import LSTMPipeline

if __name__ == '__main__':
  pipeline = LSTMPipeline()
  pipeline.train_model(epochs=15, batch_size=128)
  pipeline.save_weights('models/lstm_weights.pth')