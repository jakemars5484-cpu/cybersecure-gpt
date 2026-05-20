import torch
import torch.nn as nn
import torch.optim as optim
import json
import random

# ----------------------------
# DATASET
# ----------------------------

class Dataset:
    def __init__(self, path="dataset.json"):
        with open(path, "r") as f:
            self.data = json.load(f)

    def get_batches(self, batch_size=8):
        random.shuffle(self.data)
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i+batch_size]


# ----------------------------
# TOKENIZER (simple char-level)
# ----------------------------

class CharTokenizer:
    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")
        self.stoi = {c:i+1 for i,c in enumerate(chars)}
        self.itos = {i+1:c for i,c in enumerate(chars)}
        self.vocab_size = len(chars) + 1

    def encode(self, text, max_len=64):
        x = [self.stoi.get(c, 0) for c in text.lower()]
        x = x[:max_len]
        x += [0] * (max_len - len(x))
        return torch.tensor(x)

    def decode(self, x):
        return "".join([self.itos.get(i, "") for i in x if i != 0])


# ----------------------------
# TINY GPT MODEL
# ----------------------------

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(64, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4),
            num_layers=2
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T).unsqueeze(0).to(x.device)

        x = self.emb(x) + self.pos(pos)

        x = self.transformer(x)
        return self.fc(x)


# ----------------------------
# REWARD MODEL (RLHF CORE)
# ----------------------------

class RewardModel(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.emb(x).mean(dim=1)
        return self.fc(x).squeeze()


# ----------------------------
# GENERATION
# ----------------------------

def generate(model, tok, prompt, max_len=40):
    model.eval()

    x = tok.encode(prompt).unsqueeze(0)

    for _ in range(max_len):
        logits = model(x)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, 1)

        x = torch.cat([x, next_id], dim=1)

    return tok.decode(x[0].tolist())


# ----------------------------
# TRAIN REWARD MODEL (your RLHF step)
# ----------------------------

def train_reward(model, reward_model, dataset, tok, epochs=5):
    opt = optim.Adam(reward_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total = 0

        for batch in dataset.get_batches():

            texts = []
            labels = []

            for item in batch:
                text = item["prompt"] + " " + item["response"]
                score = item.get("score", 0)

                texts.append(tok.encode(text))
                labels.append(torch.tensor(score, dtype=torch.float32))

            x = torch.stack(texts)
            y = torch.stack(labels)

            pred = reward_model(x)

            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print("reward epoch:", epoch, "loss:", total)


# ----------------------------
# TRAIN GPT (simple imitation learning)
# ----------------------------

def train_gpt(model, dataset, tok, epochs=5):
    opt = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0

        for batch in dataset.get_batches():

            x_list = []
            y_list = []

            for item in batch:
                text = item["prompt"] + item["response"]

                x = tok.encode(text)
                y = tok.encode(text)

                x_list.append(x)
                y_list.append(y)

            x = torch.stack(x_list)
            y = torch.stack(y_list)

            logits = model(x)

            loss = loss_fn(
                logits.view(-1, tok.vocab_size),
                y.view(-1)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print("gpt epoch:", epoch, "loss:", total)


# ----------------------------
# MAIN
# ----------------------------

def main():
    tok = CharTokenizer()

    dataset = Dataset("dataset.json")

    gpt = TinyGPT(tok.vocab_size)
    reward_model = RewardModel(tok.vocab_size)

    print("Training GPT...")
    train_gpt(gpt, dataset, tok)

    print("Training Reward Model (RLHF)...")
    train_reward(reward_model, dataset, tok)

    # test
    while True:
        prompt = input("\nYou: ")
        out = generate(gpt, tok, prompt)
        print("AI:", out)


if __name__ == "__main__":
    main()