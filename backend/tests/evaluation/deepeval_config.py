"""DeepEval configuration for using Google Gemini models.

This module configures DeepEval to use Google Gemini 2.5 Flash using DeepEval's
built-in GeminiModel class, leveraging the existing Gemini configuration from the project.

"""

from typing import Optional
from deepeval.models import GeminiModel, DeepSeekModel

from app.config import settings


def get_gemini_evaluation_model(
    temperature: float = 0.0,
    model_name: str | None = None
) -> GeminiModel:
    """Get a configured Gemini model for DeepEval evaluation metrics.
    
    Uses DeepEval's built-in GeminiModel class with your existing project configuration.
    Following official DeepEval documentation for GeminiModel configuration.
    
    Args:
        temperature: Temperature for generation (default: 0.0 for deterministic)
        model_name: Optional model name override (defaults to settings.GEMINI_MODEL)
        
    Returns:
        GeminiModel: Configured Gemini model for evaluation
        
    Raises:
        ValueError: If GOOGLE_GEMINI_API_KEY is not configured
        
    """
    if not settings.GOOGLE_GEMINI_API_KEY:
        raise ValueError(
            "GOOGLE_GEMINI_API_KEY not configured. "
            "Please set it in your .env file to use Gemini for evaluation."
        )
    
    # Use DeepEval's built-in GeminiModel (following official docs)
    return GeminiModel(
        model_name=model_name or settings.GEMINI_MODEL,
        api_key=settings.GOOGLE_GEMINI_API_KEY,
        temperature=temperature
    )


def validate_evaluation_config() -> bool:
    """Validate that the evaluation configuration is correct.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not settings.GOOGLE_GEMINI_API_KEY:
        raise ValueError(
            "GOOGLE_GEMINI_API_KEY not configured. "
            "Add it to your .env file to use Gemini for evaluation."
        )
    
    if not settings.GEMINI_MODEL:
        raise ValueError(
            "GEMINI_MODEL not configured. "
            "Check app/config.py for model configuration."
        )
    
    # Try to create the model to verify it works
    try:
        model = get_gemini_evaluation_model()
        # Test a simple generation
        response = model.generate("Say hello")
        if response:
            return True
        raise ValueError("Model returned empty response")
    except Exception as e:
        raise ValueError(
            f"Failed to load Gemini model for evaluation: {e}"
        )

