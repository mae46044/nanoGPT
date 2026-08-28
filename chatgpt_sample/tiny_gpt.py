import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基本設定
# ============================================================

torch.manual_seed(0)

text = "hello world"

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


print("=== Vocabulary ===")
print(chars)
print("vocab_size =", len(chars))
print()


# ============================================================
# 2. Tiny GPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_model = d_model

        # token ID -> vector
        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Transformer block
        self.ln1 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # hidden vector -> vocabulary logits
        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, token_ids, debug=False):

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        X = self.embedding(token_ids)

        if debug:
            print("Embedding")
            print("X.shape =", X.shape)
            print(X)
            print()

        T = X.shape[0]

        # ----------------------------------------------------
        # Self Attention
        # ----------------------------------------------------

        Y = self.ln1(X)

        Q = self.q_proj(Y)
        K = self.k_proj(Y)
        V = self.v_proj(Y)

        if debug:
            print("Q/K/V")
            print("Q.shape =", Q.shape)
            print("K.shape =", K.shape)
            print("V.shape =", V.shape)
            print()

        # Q @ K^T
        scores = Q @ K.transpose(-2, -1)

        # scale
        scores = scores / math.sqrt(self.d_model)

        if debug:
            print("Attention scores before mask")
            print("scores.shape =", scores.shape)
            print(scores)
            print()

        # ----------------------------------------------------
        # Causal mask
        #
        # future token を見えなくする
        # ----------------------------------------------------

        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=X.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            mask,
            float("-inf")
        )

        if debug:
            print("Attention scores after mask")
            print(scores)
            print()

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        weights = F.softmax(
            scores,
            dim=-1
        )

        if debug:
            print("Attention weights")
            print(weights)
            print()

            print("row sums")
            print(weights.sum(dim=-1))
            print()

        # ----------------------------------------------------
        # weighted sum of V
        # ----------------------------------------------------

        A = weights @ V

        if debug:
            print("Attention output")
            print("A.shape =", A.shape)
            print(A)
            print()

        # Residual connection
        X = X + A

        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        M = self.mlp(
            self.ln2(X)
        )

        if debug:
            print("MLP output")
            print("M.shape =", M.shape)
            print(M)
            print()

        # Residual connection
        X = X + M

        # ----------------------------------------------------
        # logits
        # ----------------------------------------------------

        logits = self.lm_head(X)

        if debug:
            print("Logits")
            print("logits.shape =", logits.shape)
            print(logits)
            print()

        return logits


# ============================================================
# 3. モデル生成
# ============================================================

vocab_size = len(chars)
d_model = 8

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=d_model
)

print("=== Model ===")
print(model)
print()


# ============================================================
# 4. forward の中身を見る
# ============================================================

sample = "hello"

x = torch.tensor(
    encode(sample),
    dtype=torch.long
)

print("=== Input ===")
print("text =", sample)
print("token IDs =", x)
print()

print("=== Forward debug ===")

with torch.no_grad():
    logits = model(
        x,
        debug=True
    )


# ============================================================
# 5. 学習データ
#
# hello world
#
# input:
# h e l l o   w o r l
#
# target:
# e l l o   w o r l d
# ============================================================

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

train_x = data[:-1]
train_y = data[1:]

print("=== Training data ===")
print("input IDs :", train_x)
print("target IDs:", train_y)

print(
    "input text :",
    decode(train_x.tolist())
)

print(
    "target text:",
    decode(train_y.tolist())
)

print()


# ============================================================
# 6. 学習前の Loss
# ============================================================

logits = model(train_x)

loss = F.cross_entropy(
    logits,
    train_y
)

print("=== Initial loss ===")
print(loss.item())
print()


# ============================================================
# 7. 学習
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)

print("=== Training ===")

for step in range(1001):

    logits = model(train_x)

    loss = F.cross_entropy(
        logits,
        train_y
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(
            f"step={step:4d}",
            f"loss={loss.item():.6f}"
        )

print()


# ============================================================
# 8. 次 token の確率を見る
# ============================================================

prompt = "hell"

prompt_ids = torch.tensor(
    encode(prompt),
    dtype=torch.long
)

with torch.no_grad():

    logits = model(prompt_ids)

    last_logits = logits[-1]

    probs = F.softmax(
        last_logits,
        dim=-1
    )

print("=== Next token probabilities ===")
print("prompt =", repr(prompt))

for i, p in enumerate(probs):

    print(
        repr(itos[i]),
        f"{p.item():.6f}"
    )

print()


# ============================================================
# 9. 文章生成
# ============================================================

def generate(
    model,
    prompt,
    max_new_tokens=20
):

    ids = encode(prompt)

    for _ in range(max_new_tokens):

        x = torch.tensor(
            ids,
            dtype=torch.long
        )

        with torch.no_grad():

            logits = model(x)

            last_logits = logits[-1]

            probs = F.softmax(
                last_logits,
                dim=-1
            )

        # 確率に従ってサンプリング
        next_id = torch.multinomial(
            probs,
            num_samples=1
        ).item()

        ids.append(next_id)

    return decode(ids)


print("=== Generation ===")

result = generate(
    model,
    prompt="h",
    max_new_tokens=30
)

print(result)

