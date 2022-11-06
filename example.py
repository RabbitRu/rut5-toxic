from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm

max_length = 256

tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=max_length)
model = T5ForConditionalGeneration.from_pretrained("t5-small")

with open('igor.txt', 'r', encoding='utf8')as fp:
    igor_txt = fp.readlines()
# encode the inputs
input_sequences = igor_txt[-90:]
output_sequences = igor_txt[:90]

# encode the inputs
print(len(igor_txt))

encoding = tokenizer(
    input_sequences,
    padding="longest",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# encode the targets
target_encoding = tokenizer(
    output_sequences,
    padding="longest",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
)
labels = target_encoding.input_ids

# replace padding token id's of the labels by -100 so it's ignored by the loss
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.item()