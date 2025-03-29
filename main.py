from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)

model.eval()

sentence = "שלום"
dummy_input = tokenizer('שלום', return_tensors='pt')


np_input = {k: v.numpy() for k, v in dummy_input.items()}
np.savez("dummy_input.npz", **np_input)

input_names = ['input_ids', 'token_type_ids', 'attention_mask']
output_names = ['nikud_logits', 'shin_logits']

torch.onnx.export(
    model,
    args=(dummy_input['input_ids'], dummy_input['token_type_ids'], dummy_input['attention_mask']),
    f="model.onnx",
    input_names=['input_ids', 'token_type_ids', 'attention_mask'],
    output_names=['nikud_logits', 'shin_logits'],
    do_constant_folding=True,
    opset_version=14,
)