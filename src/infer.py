import onnxruntime as ort
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)

model.eval()

# Prepare input for tokenization
sentence = 'שלום וברכה'
dummy_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

# Load ONNX model
onnx_model = ort.InferenceSession("model.onnx")

# Prepare the tokenized input in NumPy format for inference
onnx_inputs = {
    "input_ids": dummy_input['input_ids'].numpy(),
    "token_type_ids": dummy_input['token_type_ids'].numpy(),
    "attention_mask": dummy_input['attention_mask'].numpy(),
}

# Run inference
outputs = onnx_model.run(["nikud_logits", "shin_logits"], onnx_inputs)

# Process output
nikud_logits, shin_logits = outputs

# Convert logits to predicted labels (indices of highest probability)
nikud_predictions = np.argmax(nikud_logits, axis=-1)
shin_predictions = np.argmax(shin_logits, axis=-1)

# Map predictions back to corresponding tokens in the vocabulary (optional)
nikud_tokens = tokenizer.convert_ids_to_tokens(nikud_predictions[0])
shin_tokens = tokenizer.convert_ids_to_tokens(shin_predictions[0])

# Print the results
print("Nikud Predictions:", nikud_tokens)
print("Shin Predictions:", shin_tokens)
