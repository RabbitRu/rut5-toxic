import numpy as np
import pandas as pd
from tqdm import tqdm

with open('igor.txt', 'r', encoding='utf8')as fp:
    igor_txt = fp.readlines()

max_length = 0
max_len_example = ''
print(igor_txt[0])
for i in tqdm(igor_txt):
    max_length = len(i) if len(i) > max_length else max_length
    max_len_example = i if len(i) > max_length else max_len_example

print(max_length)
print(max_len_example)

with open('data.txt', 'r', encoding='utf8')as fp:
    igor_txt = fp.readlines()

max_length = 0
print(igor_txt[0])
for i in tqdm(igor_txt):
    max_length = len(i) if len(i) > max_length else max_length
    max_len_example = i if len(i) > max_length else max_len_example

print(max_length)
print(max_len_example)