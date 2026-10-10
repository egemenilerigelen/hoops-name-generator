from fastapi.testclient import TestClient
import pytest

from app import app


@pytest.fixture(scope='module')
def client():
  with TestClient(app) as c:
    yield c


def test_index_serves_frontend(client):
  response = client.get('/')
  assert response.status_code == 200
  assert 'text/html' in response.headers['content-type']


@pytest.mark.parametrize('asset', ['name-generator.js', 'model.json'])
def test_browser_model_assets_are_served(client, asset):
  assert client.get(f'/{asset}').status_code == 200


def test_generate_default_temperature(client):
  response = client.get('/api/generate')
  assert response.status_code == 200
  body = response.json()
  assert body['temperature'] == 0.7
  assert len(body['name'].split(' ')) >= 2


def test_generate_accepts_slider_range(client):
  for temperature in (0.2, 1.2):
    assert client.get(f'/api/generate?temperature={temperature}').status_code == 200


def test_generate_rejects_invalid_temperature(client):
  for temperature in ('0', '-1', '9', 'abc'):
    assert client.get(f'/api/generate?temperature={temperature}').status_code == 422


def test_health(client):
  assert client.get('/api/health').json() == {'status': 'ok'}
