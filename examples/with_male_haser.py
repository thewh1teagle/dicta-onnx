from dicta_onnx import Dicta

dicta = Dicta('./model.onnx')
sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
with_diacritics = dicta.add_diacritics(sentence, '|')
male = dicta.get_niqqud_male(with_diacritics, '|')
haser = dicta.get_niqqud_haser(with_diacritics, '|')
print('Sentence:', sentence)
print('With diacritics:', with_diacritics)
print('With niqqud male:', male)
print('With niqqud haser:', haser)