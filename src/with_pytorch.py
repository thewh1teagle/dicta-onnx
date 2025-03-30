from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)

model.eval()

sentence = 'שלום וברכה'
result = model.predict([sentence], tokenizer)
print(result)