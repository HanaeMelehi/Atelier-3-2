from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

class CustomDataset(Dataset):
    def _init_(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        return torch.tensor(self.tokenizer.encode(self.data[idx]))

# Instantiate the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Example of a placeholder dataset
custom_dataset = ["Your custom sentence here"] * 100

# Define the model configuration
config = GPT2Config.from_pretrained('gpt2')

# Fine-tune the model
fine_tuned_model = GPT2LMHeadModel(config=config)
fine_tuned_model.train()

train_dataset = CustomDataset(custom_dataset, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Example fine-tuning loop
optimizer = torch.optim.AdamW(fine_tuned_model.parameters(), lr=5e-5)

for epoch in range(3):  
    for batch in train_dataloader:
        inputs = batch 
        labels = batch  

        outputs = fine_tuned_model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def generate_paragraph(prompt, model, tokenizer, max_length=200, temperature=0.8):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example prompt
prompt = "In a galaxy far, far away"

# Generate a new paragraph
generated_paragraph = generate_paragraph(prompt, fine_tuned_model, tokenizer)
print(generated_paragraph)