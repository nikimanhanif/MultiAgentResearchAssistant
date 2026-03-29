import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Patch the persistence initializers before importing main so they don't actually hit the db on import/test
with patch("app.main.initialize_checkpointer", new=AsyncMock()) as mock_init_c, \
     patch("app.main.initialize_store", new=AsyncMock()) as mock_init_s, \
     patch("app.main.shutdown_checkpointer", new=AsyncMock()) as mock_shut_c, \
     patch("app.main.shutdown_store", new=AsyncMock()) as mock_shut_s:
    
    from app.main import app

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "Multi-Agent Research Assistant API" in response.json()["message"]

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_lifespan_events():
    # Calling TestClient as a context manager triggers lifespan startup and shutdown
    with patch("app.main.initialize_checkpointer", new_callable=AsyncMock) as mock_init_c, \
         patch("app.main.initialize_store", new_callable=AsyncMock) as mock_init_s, \
         patch("app.main.shutdown_checkpointer", new_callable=AsyncMock) as mock_shut_c, \
         patch("app.main.shutdown_store", new_callable=AsyncMock) as mock_shut_s:
        
        with TestClient(app) as client:
            mock_init_c.assert_called_once()
            mock_init_s.assert_called_once()
            
            # Shutdown shouldn't be called yet while inside context block
            mock_shut_c.assert_not_called()
            mock_shut_s.assert_not_called()
            
            client.get("/health")
        
        # Now context is exited, shutdown should be called
        mock_shut_c.assert_called_once()
        mock_shut_s.assert_called_once()
