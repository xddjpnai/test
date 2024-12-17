from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name: str):
    """
    Загружает модель и токенизатор на основе имени модели.

    Args:
        model_name (str): Название модели для загрузки.

    Returns:
        model: Загруженная модель.
        tokenizer: Соответствующий токенизатор.
    """
    try:
        logger.info(f"Загрузка модели и токенизатора: {model_name}")
        
        # Загрузка модели и токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        logger.info(f"Модель {model_name} успешно загружена")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели {model_name}: {e}")
        raise
