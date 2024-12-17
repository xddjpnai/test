import logging
import os

def setup_logging(log_dir="./logs", log_filename="training.log"):
    """
    Настраивает логирование с сохранением логов в файл и выводом на консоль.

    Args:
        log_dir (str): Путь к папке для сохранения логов.
        log_filename (str): Имя файла с логами.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Логи сохраняются в файл
            logging.StreamHandler()  # Логи выводятся в консоль
        ]
    )
