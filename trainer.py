from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import logging
from utils import preprocess_text

logger = logging.getLogger(__name__)

def train_model(model, tokenizer, dataset, method_config):
    """
    Обучает модель с заданным методом (freeze, full fine-tune, LoRA).
    
    Args:
        model: Модель для обучения.
        tokenizer: Токенизатор модели.
        dataset: Данные для обучения.
        method_config (dict): Конфигурация метода.

    Returns:
        model: Обученная модель.
    """
    try:
        # Токенизация данных
        tokenized_dataset = dataset.map(lambda x: preprocess_text(x, tokenizer, 
                                                                  method_config["max_input_length"],
                                                                  method_config["max_target_length"]), 
                                        batched=True)
        
        # Настройка метода
        if method_config["type"] == "lora":
            logger.info("Применение LoRA адаптации")
            lora_config = LoraConfig(
                r=method_config["rank"],
                lora_alpha=method_config["lora_alpha"],
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            model = get_peft_model(model, lora_config)

        elif method_config["type"] == "freeze":
            logger.info("Замораживание всех параметров модели")
            for param in model.parameters():
                param.requires_grad = False

        # Параметры обучения
        training_args = TrainingArguments(
            output_dir=method_config["output_dir"],
            evaluation_strategy="epoch",
            learning_rate=method_config["learning_rate"],
            per_device_train_batch_size=method_config["batch_size"],
            num_train_epochs=method_config["num_epochs"],
            logging_dir=f"{method_config['output_dir']}/logs",
            logging_steps=method_config["logging_steps"]
        )

        # Обучение модели
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )
        logger.info("Начало обучения...")
        trainer.train()
        logger.info("Обучение завершено.")
        return model

    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        raise