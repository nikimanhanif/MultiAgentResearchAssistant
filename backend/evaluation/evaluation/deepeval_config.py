"""DeepEval configuration for OpenAI gpt-5-nano evaluation model.

This module configures DeepEval to use OpenAI's gpt-5-nano for evaluation metrics,
following DeepEval's native OpenAI integration documentation.
"""

from typing import Optional
from deepeval.models import GPTModel

from app.config import settings


def get_evaluation_model(
    temperature: float = 0.0,
    model_name: str | None = None
) -> GPTModel:
    """Get OpenAI gpt-5-nano for DeepEval evaluation metrics.
    
    Uses OpenAI's gpt-5-nano model for all evaluation metrics.
    
    Args:
        temperature: Temperature for evaluation (default: 0.0 for deterministic)
        model_name: Optional model override (defaults to settings.OPENAI_EVAL_MODEL)
        
    Returns:
        GPTModel: Configured OpenAI model for evaluation
        
    Raises:
        ValueError: If OPENAI_API_KEY is not configured
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. "
            "Please set it in your .env file for evaluation."
        )
    
    return GPTModel(
        model=model_name or settings.OPENAI_EVAL_MODEL,
        _openai_api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
        cost_per_input_token=0.00000005,
        cost_per_output_token=0.0000004
    )


def validate_evaluation_config() -> bool:
    """Validate evaluation configuration.
    
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. "
            "Add it to your .env file for evaluation."
        )
    
    if not settings.OPENAI_EVAL_MODEL:
        raise ValueError(
            "OPENAI_EVAL_MODEL not configured in app/config.py"
        )
    
    # Test model initialization
    try:
        model = get_evaluation_model()
        # Simple test generation
        response = model.generate("Say hello")
        if response:
            return True
        raise ValueError("Model returned empty response")
    except Exception as e:
        raise ValueError(f"Failed to initialize evaluation model: {e}")
