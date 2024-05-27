import torch
from hparams import p, frac_train, seed
import random

def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = torch.tensor([(i, j, num) for i in range(num) for j in range(num)])
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]

train, test = gen_train_test(frac_train, p, seed)
print(len(train), len(test))
