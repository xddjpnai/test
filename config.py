import os

class Config:
    # Пути и данные
    dataset_name = "squad"  # Пример названия датасета
    output_dir = "./results"  # Папка для сохранения результатов

    # Список моделей для сравнения
    model_names = [
    "google/flan-t5-small", 
    "google/flan-t5-base", 
    "google/flan-t5-large",
    ]

    # Методы обучения
    methods = {
        "freeze": {
            "type": "freeze",
            "learning_rate": 5e-5,
            "batch_size": 4,
            "num_epochs": 1,
            "max_input_length": 128,
            "max_target_length": 128,
            "output_dir": os.path.join(output_dir, "freeze"),
            "logging_steps": 50
        },
        "full_finetune": {
            "type": "full_finetune",
            "learning_rate": 5e-5,
            "batch_size": 4,
            "num_epochs": 1,
            "max_input_length": 128,
            "max_target_length": 128,
            "output_dir": os.path.join(output_dir, "finetune"),
            "logging_steps": 50
        },
        "lora_rank_4": {
            "type": "lora",
            "rank": 4,
            "lora_alpha": 16,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "num_epochs": 1,
            "max_input_length": 128,
            "max_target_length": 128,
            "output_dir": os.path.join(output_dir, "lora_rank_4"),
            "logging_steps": 50
        },
        "lora_rank_8": {
            "type": "lora",
            "rank": 8,
            "lora_alpha": 32,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "num_epochs": 1,
            "max_input_length": 128,
            "max_target_length": 128,
            "output_dir": os.path.join(output_dir, "lora_rank_8"),
            "logging_steps": 50
        },
        "lora_rank_16": {
            "type": "lora",
            "rank": 16,
            "lora_alpha": 64,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "num_epochs": 1,
            "max_input_length": 128,
            "max_target_length": 128,
            "output_dir": os.path.join(output_dir, "lora_rank_16"),
            "logging_steps": 50
        }
    }
