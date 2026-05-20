# 🧠 Manual RLHF TinyGPT

A simple PyTorch-based experiment for building and training a small GPT-style model using manual reinforcement learning from human feedback (RLHF) with 👍 / 😐 / 👎 ratings.

This project demonstrates how AI systems can learn behavior from human preference signals.

---

## 🚀 Features

- 🧠 Tiny Transformer GPT (PyTorch)
- 🧮 Reward model (learns your preferences)
- 👍 😐 👎 manual feedback training
- 💬 Text generation loop
- 💾 JSON dataset storage
- 🔁 Train + test in one script
- 🧨 Safe dataset reset tool

---

## 📁 Project Structure


dataset.json # Training data (prompt, response, score)
main.py # GPT model + reward model + training loop
reset.py # Safely resets dataset.json
README.md # Project documentation


---

## 🧠 How It Works

### 1. Data Collection
You interact with the AI and rate outputs:

```json
{
  "prompt": "hello",
  "response": "hi there!",
  "score": 1
}
Ratings:
👍 = 1 (good)
😐 = 0 (neutral)
👎 = -1 (bad)
2. Training Process

The system trains two models:

🧠 GPT Model

Learns to generate text.

🧮 Reward Model

Learns what YOU consider good or bad responses.

3. Learning Loop
GPT generates a response
You rate it
Data is saved
Models are trained on your feedback
⚙️ Installation
pip install torch
▶️ Run the System
python main.py
🧨 Reset Dataset

To wipe all training data:

python reset.py

You will see:

Are you sure you want to reset? (y/n)

Type:

y → deletes dataset
n → cancels safely
🧪 Example Usage
You: hello
AI: hi there!
Rate: 👍 / 😐 / 👎
🧠 What This Project Teaches
How language models learn patterns
How RLHF (Reinforcement Learning from Human Feedback) works
How reward signals shape AI behavior
How datasets influence intelligence
⚠️ Limitations

This is a learning system only:

Not production-level AI
Small model capacity
Simplified tokenization
No large-scale training infrastructure
🔥 Future Improvements
Add real tokenizer (BPE / SentencePiece)
Upgrade to GPT-2 or LLaMA fine-tuning
Add PPO reinforcement learning
Add curses-based training UI
Add VM sandbox tool execution system
Add multi-user feedback training
📜 License

This project is free to use for learning, experimentation, and research.