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
n_embd=32
#-------------------
torch.manual_seed(1337)


#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
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

class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key=nn.Linear(n_embd,head_size)
    self.query=nn.Linear(n_embd,head_size)
    self.value=nn.Linear(n_embd,head_size)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
  def forward(self,x):
    B,T,C=x.shape
    k=self.key(x)
    q=self.query(x)
    wei=q@k.transpose(-2,-1)*C**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei=F.softmax(wei,dim=-1)
    v=self.value(x)
    out=wei@v
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
      super().__init__()
      self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj=nn.Linear(n_embd,n_embd)
    def forward(self,x):
      out=torch.cat([h(x) for h in self.heads],dim=-1)
      out=self.proj=self.proj(out)
      return out

class FeedForward(nn.Module):
    def __init__(self,n_embd):
      super().__init__() 
      self.net=nn.Sequential(
         nn.Linear(n_embd,4*n_embd),
         nn.ReLU(),
         nn.Linear(4*n_embd,n_embd),
      )
    def forward(self,x):
      return self.net(x)
    


class Block(nn.Module):
  def __init__(self,n_embd,n_head):
    super().__init__()
    head_size=n_embd//n_head
    self.sa=MultiHeadAttention(n_head,head_size)
    self.ffwd=FeedForward(n_embd)
    self.ln1=nn.LayerNorm(n_embd)
    self.ln2=nn.LayerNorm(n_embd)
  def forward(self,x):
    x=x+self.sa(self.ln1(x))
    x=x+self.ffwd(self.ln2(x))
    return x 


class BigramModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding=nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table=nn.Embedding(block_size,n_embd)
    self.blocks=nn.Sequential(
       Block(n_embd,n_head=4),
       Block(n_embd,n_head=4),
       Block(n_embd,n_head=4),
       nn.LayerNorm(n_embd),

    )
    self.lm_head=nn.Linear(n_embd,vocab_size)

  def forward(self, idx, targets=None):
      B,T=idx.shape
      # idx and targets are both (B,T) tensor of integers
      token_emb = self.token_embedding(idx) # (B,T,C)
      pos_emb=self.position_embedding_table(torch.arange(T))
      x=token_emb+pos_emb
      x=self.blocks(x)
      logits=self.lm_head(x)
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
        idx_cond = idx[:, -block_size:]
        # get the predictions

        logits, loss = self(idx_cond)
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