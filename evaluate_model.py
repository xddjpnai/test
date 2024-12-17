from evaluate import load
from transformers import pipeline
from utils import preprocess_text
import logging
import numpy as np

logger = logging.getLogger(__name__)

def evaluate_model(model, tokenizer, dataset, method_config):
    """
    Оценка модели по метрикам F1, Exact Match и BLEU.
    
    Args:
        model: Обученная модель.
        tokenizer: Токенизатор модели.
        dataset: Данные для оценки.
        method_config (dict): Конфигурация метода.

    Returns:
        dict: Результаты метрик.
    """
    try:
        # Подготовка пайплайна
        text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        predictions, references = [], []

        for example in dataset:
            input_text = example["input_text"]
            reference_text = example["reference_text"]
            output = text_generator(input_text, max_length=method_config["max_target_length"])[0]["generated_text"]
            predictions.append(output)
            references.append(reference_text)

        # Метрики
        em = np.mean([1 if pred.strip() == ref.strip() else 0 for pred, ref in zip(predictions, references)])
        f1 = np.mean([len(set(pred.split()) & set(ref.split())) / len(set(pred.split())) for pred, ref in zip(predictions, references)])
        bleu = load("bleu").compute(predictions=predictions, references=[[ref] for ref in references])
        
        results = {
            "Exact Match (EM)": em,
            "F1 Score": f1,
            "BLEU": bleu["bleu"]
        }
        logger.info(f"Результаты оценки: {results}")
        return results

    except Exception as e:
        logger.error(f"Ошибка при оценке: {e}")
        raise