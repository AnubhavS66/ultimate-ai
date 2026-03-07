import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# ---------------- CONFIG ---------------- #
batch_size = 32
block_size = 256 
max_iters = 2000 
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
temperature = 0.8
top_k = 40

print(f"Running on: {device}")

# ---------------- DATA ---------------- #
def load_data():
    if not os.path.exists("input.txt") or os.path.getsize("input.txt") < 100:
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write("User: Hello\nAI: Hi! How can I help?\nUser: Who are you?\nAI: I am an AI.\n" * 1000)

    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, encode, decode

train_data, val_data, vocab_size, encode, decode = load_data()

# ---------------- MODEL ---------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * ((C // n_head) ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (self.attn_drop(att) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.attn = RMSNorm(n_embd), Attention()
        self.ln2, self.ff = RMSNorm(n_embd), nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))

class UltimateAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Positional embedding logic fixed for sliding window
        positions = torch.arange(T, device=device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Correct way to get the newline token ID
        nl_token = encode('\n')
        stop_id = nl_token[0] if nl_token else -1
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # Sliding window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1, None]] = float('-inf')
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == stop_id: break
        return idx

# ---------------- EXECUTION ---------------- #
model = UltimateAI().to(device)

def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Training started...")
    for i in range(max_iters):
        # Simple batch fetching
        max_idx = len(train_data) - block_size - 1
        ix = torch.randint(0, max_idx, (batch_size,))
        xb = torch.stack([train_data[j:j+block_size] for j in ix]).to(device)
        yb = torch.stack([train_data[j+1:j+block_size+1] for j in ix]).to(device)
        
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % eval_interval == 0: print(f"Step {i}: Loss {loss.item():.4f}")
    torch.save(model.state_dict(), "model.pth")
    print("Training Done!")

def chat():
    if os.path.exists("model.pth"): 
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Model Loaded!")
    model.eval()
    print("\nAI Ready! (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit': break
        
        # Build prompt and crop to block_size
        prompt = "User: " + user_input + "\nAI:"
        encoded_prompt = encode(prompt)
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)[:, -block_size:]
        
        # Generate
        generated_ids = model.generate(context, 100)
        
        # Get only the new tokens (after the prompt)
        new_tokens = generated_ids[0, context.size(1):].tolist()
        response = decode(new_tokens)
        print("AI:", response.strip())

if __name__ == "__main__":
    # Pehle train karega (agar pehli baar hai), phir chat
    if not os.path.exists("model.pth"):
        train()
    chat()
