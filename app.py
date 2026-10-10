from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import CHECKPOINT_PATH, DEFAULT_TEMPERATURE, MAX_TEMPERATURE, MIN_TEMPERATURE, STATIC_DIR
from src.generator import NameGenerationError, NameGeneratorService


class GeneratedName(BaseModel):
  name: str
  temperature: float


def get_generator(request: Request) -> NameGeneratorService:
  """Endpoint'ler üreticiyi bu bağımlılık üzerinden alır; testlerde kolayca değiştirilebilir."""
  return request.app.state.generator


def create_app(checkpoint_path: Path = CHECKPOINT_PATH, static_dir: Path = STATIC_DIR) -> FastAPI:

  @asynccontextmanager
  async def lifespan(app: FastAPI):
    # Model süreç başına bir kez yüklenir; istekler sırasında disk erişimi olmaz
    app.state.generator = NameGeneratorService.from_checkpoint(checkpoint_path)
    yield

  app = FastAPI(
      title='Hoops Name Generator API',
      description='LSTM tabanlı basketbolcu ismi üretici servis',
      lifespan=lifespan,
  )

  @app.get('/api/generate', response_model=GeneratedName)
  def generate_name(
      generator: Annotated[NameGeneratorService, Depends(get_generator)],
      temperature: float = Query(
          DEFAULT_TEMPERATURE,
          ge=MIN_TEMPERATURE,
          le=MAX_TEMPERATURE,
          description='Örnekleme sıcaklığı: düşük = güvenli, yüksek = yaratıcı',
      ),
  ):
    try:
      name = generator.generate(temperature)
    except NameGenerationError as e:
      raise HTTPException(status_code=503, detail=str(e)) from e
    return GeneratedName(name=name, temperature=temperature)

  @app.get('/api/health')
  def health():
    return {'status': 'ok'}

  # Arayüz ve tarayıcı modeli (index.html, name-generator.js, model.json); API rotalarından sonra eklenmeli
  app.mount('/', StaticFiles(directory=static_dir, html=True), name='static')
  return app


app = create_app()
