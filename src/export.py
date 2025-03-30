from transformers import AutoModel, AutoTokenizer
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)

model.eval()

# Prepare input for tokenization
sentence = 'שלום וברכה'
dummy_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

# Ensure token_type_ids exists (some tokenizers do not return it)
dummy_input['token_type_ids'] = dummy_input.get('token_type_ids', torch.zeros_like(dummy_input['input_ids']))

# Debugging output to ensure correct size
print(f"Tokenized input: {dummy_input}")

# Export to ONNX
torch.onnx.export(
    model,
    args=(dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['token_type_ids']),
    f="model.onnx",
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['nikud_logits', 'shin_logits'],
    do_constant_folding=True,
    opset_version=14,
)

print("ONNX export completed.")
