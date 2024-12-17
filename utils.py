import os
import json
import torch
import logging

logger = logging.getLogger(__name__)

def save_model(model, output_dir):
    """
    Сохраняет модель в указанную директорию.

    Args:
        model: Обученная модель.
        output_dir (str): Путь к директории для сохранения.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        logger.info(f"Модель сохранена в директорию: {output_dir}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении модели: {e}")
        raise

def save_results(results, output_path):
    """
    Сохраняет результаты экспериментов в JSON-файл.

    Args:
        results (dict): Результаты экспериментов.
        output_path (str): Путь к файлу для сохранения.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info(f"Результаты сохранены в файл: {output_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов: {e}")
        raise

def check_gpu():
    """
    Проверяет доступность GPU и выводит информацию о доступном устройстве.

    Returns:
        device: Устройство ('cuda' или 'cpu').
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Доступное устройство: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

def preprocess_text(example, tokenizer, max_input_length, max_target_length):
    """
    Токенизирует входные и целевые тексты для модели.

    Args:
        example (dict): Словарь с 'input_text' и 'reference_text'.
        tokenizer: Токенизатор модели.
        max_input_length (int): Максимальная длина входного текста.
        max_target_length (int): Максимальная длина целевого текста.

    Returns:
        dict: Токенизированный текст с метками.
    """
    def preprocess_data(example):
        """Format data for QA task."""
        input_text = f"question: {example['question']} context: {example['context']}"
        target_text = example['answers']['text'][0] if example['answers']['text'] else ""
        return {'input_text': input_text, 'target_text': target_text}
    
    examples = examples.select(range(len(examples) // 10)).map(preprocess_data, remove_columns=examples.column_names)
    inputs = tokenizer(example["input_text"], max_length=max_input_length, truncation=True)
    targets = tokenizer(example["target_text"], max_length=max_target_length, truncation=True).input_ids
    inputs["labels"] = targets
    return inputs

def print_experiment_summary(models, methods):
    """
    Печатает сводку всех моделей и методов, которые будут использоваться в эксперименте.

    Args:
        models (list): Список моделей.
        methods (dict): Словарь с методами обучения.
    """
    logger.info("Сводка эксперимента:")
    logger.info("Модели для сравнения:")
    for model in models:
        logger.info(f" - {model}")

    logger.info("\nМетоды обучения:")
    for method, config in methods.items():
        logger.info(f" - {method}: {config['type']}")

