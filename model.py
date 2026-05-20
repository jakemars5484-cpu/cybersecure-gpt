import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import random
import string

CHECKPOINT_FILE = "checkpoint.pt"
BASIC_EXAMPLES = [
    ("hello", "hello"),
    ("hi", "hi"),
    ("bye", "bye"),
    ("yes", "yes means correct or okay"),
    ("no", "no means not correct or stop"),
    ("help", "help means ask for support"),
    ("python", "python is a coding language"),
    ("code", "code is instructions for a computer"),
    ("function", "a function is reusable code"),
    ("variable", "a variable stores a value"),
    ("loop", "a loop repeats code"),
    ("list", "a list stores many values"),
    ("string", "a string is text"),
    ("number", "a number is a value like 1 or 2"),
    ("print", "print shows text on the screen"),
    ("error", "an error means something went wrong"),
    ("if", "if checks a condition"),
    ("else", "else runs when if is false"),
    ("return", "return sends a value back"),
    ("add", "add means put numbers together"),
]

# ----------------------------
# DATASET
# ----------------------------

class Dataset:
    def __init__(self, path="dataset.json"):
        self.path = path

        if not os.path.exists(path):
            self.data = []
            self.save()
            return

        with open(path, "r") as f:
            self.data = json.load(f)

        if not isinstance(self.data, list):
            raise ValueError(f"{path} must contain a JSON list of training examples")

    def save(self):
        with open(self.path, "w", newline="\n") as f:
            json.dump(self.data, f, indent=2)

    def add(self, prompt, response, score):
        self.data.append({
            "prompt": prompt,
            "response": response,
            "score": score
        })
        self.save()

    def get_batches(self, batch_size=8):
        random.shuffle(self.data)
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i+batch_size]

    def add_missing(self, examples):
        existing = {
            (item.get("prompt"), item.get("response"))
            for item in self.data
        }

        added = 0
        for prompt, response in examples:
            if (prompt, response) in existing:
                continue

            self.data.append({
                "prompt": prompt,
                "response": response,
                "score": 1
            })
            added += 1

        if added:
            self.save()

        return added


# ----------------------------
# TOKENIZER (simple char-level)
# ----------------------------

class CharTokenizer:
    def __init__(self):
        chars = list(string.printable)
        self.stoi = {c:i+1 for i,c in enumerate(chars)}
        self.itos = {i+1:c for i,c in enumerate(chars)}
        self.vocab_size = len(chars) + 1

    def encode(self, text, max_len=64, pad=True):
        x = [self.stoi.get(c, 0) for c in text]
        x = x[:max_len]
        if pad:
            x += [0] * (max_len - len(x))
        return torch.tensor(x)

    def decode(self, x):
        return "".join([self.itos.get(i, "") for i in x if i != 0])


# ----------------------------
# TINY GPT MODEL
# ----------------------------

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=2
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        if T > self.max_len:
            x = x[:, -self.max_len:]
            T = self.max_len

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        x = self.emb(x) + self.pos(pos)

        x = self.transformer(x, mask=mask)
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
        return self.fc(x).squeeze(-1)


# ----------------------------
# GENERATION
# ----------------------------

def generate(model, tok, prompt, max_len=40):
    model.eval()

    x = tok.encode(prompt, max_len=model.max_len, pad=False)
    if x.numel() == 0:
        x = torch.tensor([0])
    x = x.unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x[:, -model.max_len:])
            logits[:, -1, 0] = float("-inf")
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)

            x = torch.cat([x, next_id], dim=1)

    return tok.decode(x[0].tolist())


def basic_response(prompt):
    words = prompt.lower().strip().split()
    if not words:
        return None

    meanings = dict(BASIC_EXAMPLES)
    for word in words:
        cleaned = word.strip(string.punctuation)
        if cleaned in meanings:
            return meanings[cleaned]

    return None


def parse_rating(value):
    ratings = {
        "+": 1,
        "1": 1,
        "good": 1,
        "0": 0,
        "neutral": 0,
        "-": -1,
        "-1": -1,
        "bad": -1
    }
    return ratings.get(value.strip().lower())


def save_checkpoint(path, gpt, reward_model):
    torch.save({
        "gpt": gpt.state_dict(),
        "reward_model": reward_model.state_dict()
    }, path)


def load_checkpoint(path, gpt, reward_model):
    if not os.path.exists(path):
        return False

    checkpoint = torch.load(path, map_location="cpu")
    gpt.load_state_dict(checkpoint["gpt"])
    reward_model.load_state_dict(checkpoint["reward_model"])
    return True


# ----------------------------
# TRAIN REWARD MODEL (your RLHF step)
# ----------------------------

def train_reward(reward_model, dataset, tok, epochs=5):
    if not dataset.data:
        print("reward training skipped: dataset is empty")
        return

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
    if not dataset.data:
        print("gpt training skipped: dataset is empty")
        return

    opt = optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        total = 0

        for batch in dataset.get_batches():

            x_list = []
            y_list = []

            for item in batch:
                text = item["prompt"] + " " + item["response"]
                encoded = tok.encode(text, max_len=model.max_len + 1)

                x = encoded[:-1]
                y = encoded[1:]

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
    added = dataset.add_missing(BASIC_EXAMPLES)
    if added:
        print("Added basic word examples:", added)

    gpt = TinyGPT(tok.vocab_size)
    reward_model = RewardModel(tok.vocab_size)

    if load_checkpoint(CHECKPOINT_FILE, gpt, reward_model):
        print("Loaded checkpoint.")

    print("Training GPT...")
    train_gpt(gpt, dataset, tok)

    print("Training Reward Model (RLHF)...")
    train_reward(reward_model, dataset, tok)
    save_checkpoint(CHECKPOINT_FILE, gpt, reward_model)
    print("Checkpoint saved.")

    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not prompt:
            continue

        out = basic_response(prompt) or generate(gpt, tok, prompt)
        print("AI:", out)

        rating = input("Rate (+ good / 0 neutral / - bad / skip): ")
        score = parse_rating(rating)
        if score is None:
            print("feedback skipped")
            continue

        dataset.add(prompt, out, score)
        print("feedback saved")

        print("Training on feedback...")
        train_gpt(gpt, dataset, tok, epochs=1)
        train_reward(reward_model, dataset, tok, epochs=1)
        save_checkpoint(CHECKPOINT_FILE, gpt, reward_model)
        print("Checkpoint saved.")


if __name__ == "__main__":
    main()
