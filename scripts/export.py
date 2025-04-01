# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch==2.6.0",
#     "onnx==1.17.0",
#     "onnx==4.50.3",
# ]
# ///

"""
uv run scripts/export.py
"""

import torch
import onnx
from transformers import AutoModel
from pathlib import Path

# Config
dynamic_axes = True
dynamic_axes_dict: dict | None = None
batch_size = 1
sequence_length = 128
model_name = 'dicta-il/dictabert-large-char-menaked'
output_path = 'model.onnx'

# Fetch model
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Create dummy inputs
dummy_input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long)
dummy_attention_mask = torch.ones((batch_size, sequence_length), dtype=torch.long)
dummy_token_type_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long)

# Define dynamic axes if requested
if dynamic_axes:
    dynamic_axes_dict = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
        'nikud_logits': {0: 'batch_size', 1: 'sequence_length'},
        'shin_logits': {0: 'batch_size', 1: 'sequence_length'}
    }

# Create model folder
Path(output_path).parent.mkdir(exist_ok=True)

# Define forward pass for onnx export
def forward_for_onnx(input_ids, attention_mask, token_type_ids):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True
    )
    # Return the logits separately for clear output names
    return outputs.logits.nikud_logits, outputs.logits.shin_logits

print(f"Exporting onnx model to: {output_path}...")
torch.onnx.export(
    model,
    args=(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
    f=output_path,
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['nikud_logits', 'shin_logits'],
    dynamic_axes=dynamic_axes_dict,
    opset_version=14,
    do_constant_folding=True,
    export_params=True,
    verbose=False
)
print("✅ onnx model export completed!")

# Verify the exported model
print("Verifying onnx model integrity...")
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("🎉 onnx model verification successful! Ready to use.")
