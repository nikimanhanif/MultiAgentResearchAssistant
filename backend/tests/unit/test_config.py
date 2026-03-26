import pytest
from unittest.mock import patch, MagicMock

from app.config import (
    Settings,
    get_deepseek_chat,
    get_deepseek_reasoner,
    get_deepseek_reasoner_json,
)

def test_cors_origins_list():
    """Test parsing of CORS_ORIGINS string."""
    settings = Settings(CORS_ORIGINS="http://localhost:3000, https://example.com")
    assert settings.cors_origins_list == ["http://localhost:3000", "https://example.com"]

@patch("app.config.settings")
def test_get_deepseek_chat_no_key(mock_settings):
    """Test get_deepseek_chat raises ValueError if no API key."""
    mock_settings.DEEPSEEK_API_KEY = ""
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not configured"):
        get_deepseek_chat()

@patch("app.config.settings")
@patch("langchain_deepseek.ChatDeepSeek")
def test_get_deepseek_chat_success(mock_chat, mock_settings):
    """Test get_deepseek_chat configures ChatDeepSeek correctly."""
    mock_settings.DEEPSEEK_API_KEY = "test_key"
    mock_settings.DEEPSEEK_MODEL = "test-model"
    
    get_deepseek_chat(temperature=0.8, model="override-model")
    
    mock_chat.assert_called_once_with(
        model="override-model",
        api_key="test_key",
        temperature=0.8
    )

@patch("app.config.settings")
def test_get_deepseek_reasoner_no_key(mock_settings):
    """Test get_deepseek_reasoner raises ValueError if no API key."""
    mock_settings.DEEPSEEK_API_KEY = ""
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not configured"):
        get_deepseek_reasoner()

@patch("app.config.settings")
@patch("langchain_deepseek.ChatDeepSeek")
def test_get_deepseek_reasoner_success(mock_chat, mock_settings):
    """Test get_deepseek_reasoner configures ChatDeepSeek correctly."""
    mock_settings.DEEPSEEK_API_KEY = "test_key"
    mock_settings.DEEPSEEK_REASONER_MODEL = "test-reasoner"
    
    get_deepseek_reasoner(temperature=0.1)
    
    # Assert called with default reasoner model when not overridden
    mock_chat.assert_called_once_with(
        model="test-reasoner",
        api_key="test_key",
        temperature=0.1
    )

@patch("app.config.settings")
def test_get_deepseek_reasoner_json_no_key(mock_settings):
    mock_settings.DEEPSEEK_API_KEY = ""
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY not configured"):
        get_deepseek_reasoner_json()

@patch("app.config.settings")
@patch("langchain_deepseek.ChatDeepSeek")
def test_get_deepseek_reasoner_json_success(mock_chat, mock_settings):
    mock_settings.DEEPSEEK_API_KEY = "test_key"
    mock_settings.DEEPSEEK_REASONER_MODEL = "test-reasoner"
    
    get_deepseek_reasoner_json(temperature=0.5)
    
    mock_chat.assert_called_once_with(
        model="test-reasoner",
        api_key="test_key",
        temperature=0.5,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
