from transformers import AutoModel

# Load the pre-trained model
model = AutoModel.from_pretrained("uer/gpt2-xlarge-chinese-cluecorpussmall")

# Calculate the number of parameters
model_parameters = sum(p.numel() for p in model.parameters())

print(f"The model has {model_parameters:,} parameters.")
