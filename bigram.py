import torch 
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt 
import numpy as np 



#Hyperparameters
batch_size=32 
block_size=8
num_iters=100
learning_rate=1e-2
device= "cuda" if torch.cuda.is_available() else "cpu"
#-------------------
torch.manual_seed(1337)


#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('/home/prasanjith/Desktop/transformer/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
stringtointeger={ch:i for i,ch in enumerate(chars)}
integertostring={i:ch for i,ch in enumerate(chars)}
encoder=lambda s:[stringtointeger[i] for i in s ]
decoder=lambda l:''.join([integertostring[i] for i in l])

data=torch.tensor(encoder(text),dtype=torch.long)
#splitting the data into train and valdation 
n=int(0.9*len(data))   #90 percent of the data 
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
  data=train_data if split=="train" else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))   #(low,high,(size))
  x=torch.stack([data[i:i+block_size] for i in ix])
  y=torch.stack([data[i+1:i+block_size+1] for i in ix ])
  return x,y 



class BigramModel(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,vocab_size)
  def forward(self, idx, targets=None):

      # idx and targets are both (B,T) tensor of integers
      logits = self.token_embedding(idx) # (B,T,C)
      
      if targets is None:
          loss = None
      else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C)
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits, targets)

      return logits, loss
  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # get the predictions
        logits, loss = self(idx)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx
  

m = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for step in range(1000):
  xb,yb=get_batch("train")
  logits,loss=m(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))