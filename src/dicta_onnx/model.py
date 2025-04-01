
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import re

# Constants
NIKUD_CLASSES = ['', '<MAT_LECT>', '\u05BC', '\u05B0', '\u05B1', '\u05B2', '\u05B3', '\u05B4', '\u05B5', '\u05B6', '\u05B7', '\u05B8', '\u05B9', '\u05BA', '\u05BB', '\u05BC\u05B0', '\u05BC\u05B1', '\u05BC\u05B2', '\u05BC\u05B3', '\u05BC\u05B4', '\u05BC\u05B5', '\u05BC\u05B6', '\u05BC\u05B7', '\u05BC\u05B8', '\u05BC\u05B9', '\u05BC\u05BA', '\u05BC\u05BB', '\u05C7', '\u05BC\u05C7']
SHIN_CLASSES = ['\u05C1', '\u05C2']  # shin, sin
MAT_LECT_TOKEN = '<MAT_LECT>'
MATRES_LETTERS = list('אוי')
ALEF_ORD = ord('א')
TAF_ORD = ord('ת')

def is_hebrew_letter(char):
    return ALEF_ORD <= ord(char) <= TAF_ORD

def is_matres_letter(char):
    return char in MATRES_LETTERS

nikud_pattern = re.compile(r'[\u05B0-\u05BD\u05C1\u05C2\u05C7]')
def remove_nikkud(text):
    return nikud_pattern.sub('', text)

class OnnxDiacritizationModel:
    def __init__(self, model_path, tokenizer_name='dicta-il/dictabert-large-char-menaked'):
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def predict(self, sentences, mark_matres_lectionis=None, padding='longest'):
        sentences = [remove_nikkud(sentence) for sentence in sentences]
        
        # Tokenize inputs
        inputs = self.tokenizer(sentences, padding=padding, truncation=True, return_tensors='pt', return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping').numpy()
        
        # Convert to numpy arrays for ONNX Runtime
        onnx_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy(),
            'token_type_ids': inputs['token_type_ids'].numpy() if 'token_type_ids' in inputs else np.zeros_like(inputs['input_ids'].numpy())
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, onnx_inputs)
        
        # Process outputs based on output names
        if 'nikud_logits' in self.output_names and 'shin_logits' in self.output_names:
            nikud_idx = self.output_names.index('nikud_logits')
            shin_idx = self.output_names.index('shin_logits')
            nikud_logits = outputs[nikud_idx]
            shin_logits = outputs[shin_idx]
        else:
            # Assume order is maintained as in the export
            nikud_logits, shin_logits = outputs
        
        # Get predictions
        nikud_predictions = np.argmax(nikud_logits, axis=-1)
        shin_predictions = np.argmax(shin_logits, axis=-1)
        
        ret = []
        for sent_idx, (sentence, sent_offsets) in enumerate(zip(sentences, offset_mapping)):
            # Assign the nikud to each letter
            output = []
            prev_index = 0
            for idx, offsets in enumerate(sent_offsets):
                # Add anything we missed
                if offsets[0] > prev_index:
                    output.append(sentence[prev_index:offsets[0]])
                if offsets[1] - offsets[0] != 1:
                    continue
                
                # Get next char
                char = sentence[offsets[0]:offsets[1]]
                prev_index = offsets[1]
                if not is_hebrew_letter(char):
                    output.append(char)
                    continue
                
                nikud = NIKUD_CLASSES[nikud_predictions[sent_idx][idx]]
                shin = '' if char != 'ש' else SHIN_CLASSES[shin_predictions[sent_idx][idx]]
                
                # Check for matres lectionis
                if nikud == MAT_LECT_TOKEN:
                    if not is_matres_letter(char):
                        nikud = ''  # Don't allow matres on irrelevant letters
                    elif mark_matres_lectionis is not None:
                        nikud = mark_matres_lectionis
                    else:
                        output.append(char)
                        continue
                
                output.append(char + shin + nikud)
            output.append(sentence[prev_index:])
            ret.append(''.join(output))
        
        return ret