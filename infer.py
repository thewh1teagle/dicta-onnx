import onnxruntime as ort
import numpy as np

# Load dummy input from .npz
data = np.load("dummy_input.npz")
inputs = {k: data[k] for k in data.files}
print(inputs)

# Load the ONNX model
session = ort.InferenceSession("model.onnx")

# Run inference
outputs = session.run(None, inputs)

# Unpack results
nikud_logits, shin_logits = outputs

# Print shapes
print("nikud_logits shape:", nikud_logits.shape)  # Expected: (1, seq_len, 29)
print("shin_logits shape:", shin_logits.shape)    # Expected: (1, seq_len, 2)
print(nikud_logits, shin_logits)