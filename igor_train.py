import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
from random import shuffle, randrange


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
    out_igor = fp.readlines()
with open('data.txt', 'r', encoding='utf8')as fp:
    data = fp.readlines()
# encode the inputs
prefix = 'Оскорбляй :'
divisor = int(len(data) / 2)
print(len(data))
input_sequences = [prefix + x for x in data[:divisor]]
output_sequences = data[-divisor:]

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
len_in = len(input_ids)

# encode the targets
igor_encoding = tokenizer(
    out_igor,
    padding="longest",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
)
igor_labels = igor_encoding.input_ids
igor_labels[igor_labels == tokenizer.pad_token_id] = -100

# encode the targets
out_encoding = tokenizer(
    output_sequences,
    padding="longest",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
)
out_labels = out_encoding.input_ids
out_labels[out_labels == tokenizer.pad_token_id] = -100
len_out = len(out_labels)

batch_size = 50
checkpoint = 50
for i in tqdm(range(500)):
    start_in = randrange(0, len_in - batch_size)
    shuffle(igor_labels)
    loss = model(
        input_ids=input_ids[start_in:start_in + batch_size],
        attention_mask=attention_mask[start_in:start_in + batch_size],
        labels=igor_labels[:batch_size]
    ).loss
    print(loss.item())

    start_out = randrange(0, len_out - batch_size)
    start_in = randrange(0, len_in - batch_size)
    loss = model(
        input_ids=input_ids[start_in:start_in + batch_size],
        attention_mask=attention_mask[start_in:start_in + batch_size],
        labels=out_labels[start_out:start_out + batch_size]
    ).loss
    print(loss.item())
    answer(x)

    if i % checkpoint == 0:
        model.save_pretrained(f'checkpoints\igor_{i}.pth')


model.save_pretrained('checkpoints\igor.pth')

answer(x)