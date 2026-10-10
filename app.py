from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import CHECKPOINT_PATH, DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MIN_TEMPERATURE, STATIC_DIR
from src.generator import NameGenerationError, NameGeneratorService


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Model süreç başına bir kez yüklenir; istekler sırasında disk erişimi olmaz
  app.state.generator = NameGeneratorService.from_checkpoint(CHECKPOINT_PATH)
  yield


app = FastAPI(
    title='Hoops Name Generator API',
    description='LSTM tabanlı basketbolcu ismi üretici servis',
    lifespan=lifespan,
)


class GeneratedName(BaseModel):
  name: str
  temperature: float


@app.get('/api/generate', response_model=GeneratedName)
def generate_name(
    request: Request,
    temperature: float = Query(
        DEFAULT_TEMPERATURE,
        ge=MIN_TEMPERATURE,
        le=MAX_TEMPERATURE,
        description='Örnekleme sıcaklığı: düşük = güvenli, yüksek = yaratıcı',
    ),
):
  try:
    name = request.app.state.generator.generate(temperature)
  except NameGenerationError as e:
    raise HTTPException(status_code=503, detail=str(e)) from e
  return GeneratedName(name=name, temperature=temperature)


@app.get('/api/health')
def health():
  return {'status': 'ok'}


# Arayüz ve tarayıcı modeli (index.html, name-generator.js, model.json); API rotalarından sonra eklenmeli
app.mount('/', StaticFiles(directory=STATIC_DIR, html=True), name='static')
