import fastwer

hyp = "Le Nguyen Ngoc  Thy"
ref = "Le Nguyen Ngoc Thy"

cer = fastwer.score_sent(hyp, ref, char_level=True)
print(cer)
print(len(hyp))
print(len(ref))