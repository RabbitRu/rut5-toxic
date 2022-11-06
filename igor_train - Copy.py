import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from random import shuffle


def answer(x):
    inputs = tokenizer(x, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(
            **inputs, 
            do_sample=True, top_p=0.5, num_return_sequences=3, 
            repetition_penalty=2.5,
            max_length=32,
        )
    print('Outputs:\n')
    for h in hypotheses:
        print(tokenizer.decode(h, skip_special_tokens=True))


max_length = 256
tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-small-chitchat", model_max_length=max_length)
model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-small-chitchat")


with open('igor.txt', 'r', encoding='utf8')as fp:
    igor_txt = fp.readlines()[:100]
# encode the inputs
prefix = 'Оскорбляй :'
input_sequences = [prefix + x for x in igor_txt[:int(len(igor_txt) / 2)]]
output_sequences = igor_txt[-int(len(igor_txt) / 2):]

x = prefix + 'Блядь, ебаный ты шизоид, ты можешь нормально выражать свои мысли?'

answer(x)

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

for i in tqdm(range(1000)):
    shuffle(labels)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    print(loss.item())
    answer(x)

model.save_pretrained('irog.pth')

answer(x)