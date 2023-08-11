from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch.nn.functional as F

model_name= "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier= pipeline("sentiment-analysis",model= model, tokenizer= tokenizer)
res= classifier("Gold price stretched lower and touched its weakest level in nearly a month below $1,930. The risk-averse market atmosphere and the renewed US Dollar strength drag XAU/USD lower. The benchmark 10-year US Treasury bond yield, however, is down more than 2% on the day, helping the pair limit its losses.")
print(res)
tokens= tokenizer.tokenize("Gold price stretched lower and touched its weakest level in nearly a month below $1,930. The risk-averse market atmosphere and the renewed US Dollar strength drag XAU/USD lower. The benchmark 10-year US Treasury bond yield, however, is down more than 2% on the day, helping the pair limit its losses.")
token_ids= tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer("Gold price stretched lower and touched its weakest level in nearly a month below $1,930. The risk-averse market atmosphere and the renewed US Dollar strength drag XAU/USD lower. The benchmark 10-year US Treasury bond yield, however, is down more than 2% on the day, helping the pair limit its losses.")
print(tokens)
print(token_ids)
print(input_ids)