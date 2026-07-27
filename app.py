from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from src.lstm_model import LSTMPipeline

app = FastAPI(
    title="Hoops Name Generator API",
    description="LSTM tabanlı basketbolcu ismi üretici servis",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ağırlıklarını yüklüyoruz
pipeline = LSTMPipeline(data_path="data/processed/clean_names.json")
pipeline.load_weights("models/lstm_weights.pth")


@app.get("/")
def read_root():
  # Ana adrese gelen isteklere index.html dosyasını döndürüyoruz
  return FileResponse("index.html")


@app.get("/api/generate")
def generate_name(temperature: float = 0.7):
  name = pipeline.generate_full_name(temperature=temperature)
  return {"name": name, "temperature": temperature}