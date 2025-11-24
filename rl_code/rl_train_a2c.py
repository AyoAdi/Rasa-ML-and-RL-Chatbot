# rl_train_a2c.py — Rasa 3.6+ with Actor-Critic RL
"""
RL fine-tuning for your Rasa TEDPolicy model (Behavior Cloning + A2C).
Run this from your project root:
  python3 rl_train_a2c.py
"""

import os
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import asyncio
from collections import namedtuple
from rasa.core.agent import Agent

# ----------------------------
# Config
# ----------------------------
MODEL_ARCHIVE = "models/20251106-152304-tempered-louver.tar.gz"
DOMAIN_FILE = "domain.yml"

BC_EPOCHS = 25
RL_EPISODES = 800
GAMMA = 0.99
LR = 1e-4
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLICY_SAVE = "policy_a2c_finetuned.pt"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REWARD_MAP = {
    # Greetings and casual conversation
    "greeting": +1,
    "morning": +1,
    "afternoon": +1,
    "evening": +1,
    "night": +1,
    "goodbye": +1,
    "casual": +1,

    # Positive / supportive interactions
    "thanks": +2,
    "user-agree": +1.5,
    "user-advice": +2,

    # Emotional intents — multi-step reward
    "sad": +0.8,
    "stressed": +0.8,
    "worthless": +0.8,
    "depressed": +0.8,
    "anxious": +0.8,
    "overwhelmed": +0.8,
    "lonely": +0.8,
    "hopeless": +0.8,

    # Critical intents — moderate penalties
    "suicide": -2,
    "death": -2,
    "hurt": -1.5,
    "angry": -1,
    "scared": -1,
    "worried": -0.8,

    # Fallback / system errors
    "no-response": -0.5,
    "wrong": -0.5,
    "repeat": -0.2,
    "default": -0.5,
    "neutral-response": 0,

    # Time penalty for each turn
    "time_penalty": -0.05
}

# ----------------------------
# Domain loader
# ----------------------------
def load_domain(domain_path):
    with open(domain_path, "r") as f:
        d = yaml.safe_load(f)
    actions = d.get("actions", []) or []
    intents = d.get("intents", []) or []
    responses = d.get("responses", {}) if "responses" in d else d.get("templates", {})
    return intents, actions, responses

# ----------------------------
# Actor-Critic network
# ----------------------------
class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, hidden, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        policy = torch.softmax(self.actor(h), dim=-1)
        value = self.critic(h)
        return policy, value

# ----------------------------
# State builder
# ----------------------------
def build_state_vector(intent_name, slot_dict, last_action_idx, intent_to_idx, num_slots, num_actions):
    iv = np.zeros(len(intent_to_idx), dtype=np.float32)
    if intent_name in intent_to_idx:
        iv[intent_to_idx[intent_name]] = 1.0
    sv = np.zeros(num_slots, dtype=np.float32)
    for i, (k, v) in enumerate(slot_dict.items()):
        if i >= num_slots:
            break
        sv[i] = 1.0 if v else 0.0
    la = np.zeros(num_actions, dtype=np.float32)
    if 0 <= last_action_idx < num_actions:
        la[last_action_idx] = 1.0
    return np.concatenate([iv, sv, la])

# ----------------------------
# BC dataset
# ----------------------------
Transition = namedtuple("Transition", ("state", "action_idx"))

async def collect_bc_dataset(agent, intents, actions, intent_to_idx, num_slots=10, examples_per_intent=6):
    print("Collecting BC dataset from agent...")
    dataset = []
    action_to_idx = {a: i for i, a in enumerate(actions)}
    num_actions = len(actions)
    last_action_idx = 0

    for intent in intents:
        for _ in range(examples_per_intent):
            user_text = f"[{intent}] example utterance"
            parsed = await agent.parse_message(user_text)
            parsed_intent = parsed.get("intent", {}).get("name", intent)
            slots = {}

            try:
                responses = await agent.handle_text(user_text)
                if isinstance(responses, list) and len(responses) > 0:
                    first = responses[0]
                    if isinstance(first, dict) and "text" in first:
                        out = first["text"]
                    elif isinstance(first, str):
                        out = first
                    else:
                        out = str(first)
                else:
                    out = None
            except Exception:
                out = None

            mapped_idx = 0
            if out:
                out_low = out.lower()
                for action_name in actions:
                    if action_name.startswith("utter_"):
                        key = action_name.replace("utter_", "").replace("_", " ")
                        if key in out_low:
                            mapped_idx = action_to_idx[action_name]
                            break
                if mapped_idx == 0 and "action_default_fallback" in action_to_idx:
                    mapped_idx = action_to_idx["action_default_fallback"]

            s = build_state_vector(parsed_intent, slots, last_action_idx, intent_to_idx, num_slots, num_actions)
            dataset.append(Transition(state=s, action_idx=mapped_idx))
            last_action_idx = mapped_idx

    print(f"Collected {len(dataset)} BC transitions.")
    return dataset, num_actions

# ----------------------------
# Behavior cloning
# ----------------------------
def behavior_cloning_train(policy_net, dataset, epochs=10, batch_size=32):
    policy_net.train()
    opt = optim.Adam(policy_net.parameters(), lr=LR)
    n = len(dataset)
    for ep in range(epochs):
        random.shuffle(dataset)
        losses = []
        for i in range(0, n, batch_size):
            batch = dataset[i:i + batch_size]
            states = torch.tensor(np.stack([t.state for t in batch]), device=DEVICE)
            targets = torch.tensor([t.action_idx for t in batch], device=DEVICE, dtype=torch.long)
            preds, _ = policy_net(states)
            loss = nn.functional.nll_loss(torch.log(preds + 1e-9), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"BC epoch {ep+1}/{epochs} loss: {np.mean(losses):.4f}")

# ----------------------------
# Simple User Simulator
# ----------------------------
class SimpleUserSimulator:
    def __init__(self, emotion_intents, neutral_intent="neutral-response", max_turns=8):
        self.emotions = emotion_intents
        self.neutral = neutral_intent
        self.max_turns = max_turns

    def reset(self):
        self.turn = 0
        self.current_intent = random.choice(self.emotions + [self.neutral])
        return self.current_intent, f"[{self.current_intent}] initial"

    def respond(self, bot_action_idx, actions_list):
        self.turn += 1
        action_name = actions_list[bot_action_idx] if 0 <= bot_action_idx < len(actions_list) else "unknown"
        a_lower = action_name.lower()
        if "fallback" in a_lower:
            reward = REWARD_MAP["fallback"]
            done = self.turn >= self.max_turns
            next_intent = self.current_intent
            user_text = f"[{next_intent}] fallback"
        elif any(k in a_lower for k in ["help", "advice", "meditation", "cope", "suggest"]):
            reward = REWARD_MAP["empathetic_reply"]
            if random.random() < 0.4:
                next_intent = "thanks"
                user_text = "[thanks] thank you"
                done = True
                reward += REWARD_MAP["thanks"]
            else:
                next_intent = self.neutral
                user_text = f"[{next_intent}] thinking"
                done = False
        elif any(k in a_lower for k in ["greet", "hello", "hi", "how_are_you"]):
            reward = REWARD_MAP["neutral"]
            next_intent = random.choice(self.emotions + [self.neutral])
            user_text = f"[{next_intent}] continuation"
            done = False
        else:
            reward = REWARD_MAP["neutral"]
            next_intent = random.choice(self.emotions + [self.neutral])
            user_text = f"[{next_intent}] continue"
            done = False
        reward += REWARD_MAP["time_penalty"]
        if self.turn >= self.max_turns:
            done = True
        self.current_intent = next_intent
        return next_intent, user_text, reward, done, {"action": action_name}

# ----------------------------
# A2C training
# ----------------------------
def a2c_train(policy_net, simulator, actions_list, intent_to_idx, num_slots, episodes=500):
    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    avg_rewards = []

    for ep in range(episodes):
        intent, _ = simulator.reset()
        last_action_idx = 0
        done = False
        log_probs, values, rewards = [], [], []

        while not done:
            s_vec = build_state_vector(intent, {}, last_action_idx, intent_to_idx, num_slots, len(actions_list))
            s_tensor = torch.tensor(s_vec, device=DEVICE).unsqueeze(0)
            probs, value = policy_net(s_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_idx = action.item()

            next_intent, _, r, done, _ = simulator.respond(action_idx, actions_list)

            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(r)

            last_action_idx = action_idx
            intent = next_intent

        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=DEVICE)
        values = torch.stack(values)
        advantages = returns - values

        # Actor loss
        actor_loss = (-torch.stack(log_probs) * advantages.detach()).sum()
        # Critic loss
        critic_loss = (advantages ** 2).sum()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_rewards.append(sum(rewards))
        if (ep + 1) % 50 == 0:
            print(f"[A2C] Episode {ep+1}/{episodes}, avg_reward(last50): {np.mean(avg_rewards[-50:]):.3f}")

    return avg_rewards

# ----------------------------
# MAIN
# ----------------------------
def main():
    root = os.getcwd()
    model_path = os.path.join(root, MODEL_ARCHIVE)

    print(f"📦 Loading Rasa model from: {model_path}")
    agent = Agent.load(model_path)  # synchronous load
    print("✅ Core agent loaded.")

    # Async-safe parse function
    async def parse_fn(text):
        return await agent.parse_message(text)
    agent.parse_fn = parse_fn

    # Load domain
    intents, actions, responses = load_domain(DOMAIN_FILE)
    print(f"Loaded {len(intents)} intents and {len(actions)} actions.")
    intent_to_idx = {it: i for i, it in enumerate(intents)}
    num_slots = 12

    # Collect BC dataset
    bc_dataset, num_actions = asyncio.run(
        collect_bc_dataset(agent, intents, actions, intent_to_idx, num_slots=num_slots, examples_per_intent=6)
    )

    # Build policy network
    input_dim = len(intent_to_idx) + num_slots + num_actions
    policy = ActorCriticNet(input_dim, HIDDEN, num_actions).to(DEVICE)

    # Behavior cloning
    print("🚀 Starting behavior cloning...")
    behavior_cloning_train(policy, bc_dataset, epochs=BC_EPOCHS)

    # A2C fine-tuning
    print("🎯 Starting A2C fine-tuning...")
    emotion_intents = [i for i in intents if i in {"sad", "stressed", "anxious", "depressed", "overwhelmed", "lonely", "hopeless"}]
    if not emotion_intents:
        emotion_intents = [i for i in intents if "sad" in i or "anx" in i]
    sim = SimpleUserSimulator(emotion_intents)
    rewards = a2c_train(policy, sim, actions, intent_to_idx, num_slots, episodes=RL_EPISODES)

    # Save fine-tuned policy
    torch.save({
        "state_dict": policy.state_dict(),
        "intent_to_idx": intent_to_idx,
        "actions": actions,
        "num_slots": num_slots
    }, POLICY_SAVE)
    print("✅ Saved fine-tuned policy to", POLICY_SAVE)

    # Plot reward curve
    try:
        plt.figure(figsize=(8,4))
        window = max(1, len(rewards)//50)
        smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        plt.plot(smoothed)
        plt.title("Smoothed episode reward (A2C)")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_curve_a2c.png")
        print("📊 Saved reward curve to reward_curve_a2c.png")
    except Exception as e:
        print("⚠️ Plotting failed:", e)

if __name__ == "__main__":
    main()
