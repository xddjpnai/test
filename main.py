import logging
from config import Config
from model_loader import load_model_and_tokenizer
from trainer import train_model
from evaluate_model import evaluate_model
from utils import save_model, save_results, check_gpu, print_experiment_summary
from datasets import load_dataset
from logger import setup_logging

# Настройка логирования
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Основной метод для запуска экспериментов:
    - Сравнение моделей с различными методами обучения
    """
    # Проверка устройства
    device = check_gpu()

    logger.info("Запуск экспериментов по сравнению моделей")

    # Печать сводки экспериментов
    print_experiment_summary(Config.models, Config.methods)

    # Загрузка данных
    dataset = load_dataset(Config.dataset_name, split="train[:1%]")
    logger.info("Данные успешно загружены")

    # Результаты экспериментов
    all_results = []

    # Итерация по моделям и методам
    for model_name in Config.models:
        logger.info(f"Инициализация модели: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name)

        for method_name, method_config in Config.methods.items():
            logger.info(f"Запуск метода: {method_name}")
            
            # Обучение модели
            trained_model = train_model(model, tokenizer, dataset, method_config)
            
            # Оценка модели
            metrics = evaluate_model(trained_model, tokenizer, dataset, method_config)
            logger.info(f"Метрики для {model_name} - {method_name}: {metrics}")
            
            # Сохранение результатов и модели
            save_model(trained_model, method_config["output_dir"])
            result = {
                "model": model_name,
                "method": method_name,
                "metrics": metrics
            }
            all_results.append(result)

    # Сохранение всех результатов
    save_results(all_results, "./results/experiment_results.json")
    logger.info("Эксперименты завершены. Результаты сохранены.")

if __name__ == "__main__":
    main()