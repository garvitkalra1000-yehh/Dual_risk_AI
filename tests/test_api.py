from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_summary_endpoint_available():
    resp = client.get('/summary')
    assert resp.status_code in {200, 503}


def test_worklist_endpoint_available():
    resp = client.get('/worklist')
    assert resp.status_code in {200, 503}
