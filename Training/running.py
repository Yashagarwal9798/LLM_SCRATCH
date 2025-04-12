import torch
from GPTconfig import GPT_CONFIG_124M
from GPTMODEL import GPTModel
import tiktoken
from p1 import train_loader,val_loader
from training import train_model_simple
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 8
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Save the trained model
torch.save(model.state_dict(), "model.pkl")
print("Model saved to new_gpt_model_124M.pkl")

