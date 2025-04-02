"""
pip install -U dicta-onnx

wget wget https://github.com/thewh1teagle/dicta-onnx/releases/download/model-files-v1.0/dicta-1.0.onnx
python simple.py
"""
from dicta_onnx import Dicta

dicta = Dicta('./dicta-1.0.onnx')
sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
with_diacritics = dicta.add_diacritics(sentence)
print(with_diacritics)