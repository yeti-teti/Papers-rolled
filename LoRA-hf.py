from transformers import AutoModelForCasualLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    
model = AutoModelForCasualLM.from_pretrained("", device_map="auto", trust_remote_code=True)

peft_config = LoraConfig(
    task_type=TaskType.CASUAL_LM, inference_mode=False, r=32, lora_alpha=16, lora_dropout=0.1,
    target_modules=['query_key_value']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()




