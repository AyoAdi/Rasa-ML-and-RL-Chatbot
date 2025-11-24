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

MODEL_ARCHIVE = "models/20251106-152304-tempered-louver.tar.gz"
DOMAIN_FILE = "domain.yml"

BC_EPOCHS = 25
RL_EPISODES = 800
GAMMA = 0.99
LR = 1e-4
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLICY_SAVE = "policy_finetuned.pt"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

REWARD_MAP = {
    "greeting": 2,
    "morning": 2,
    "afternoon": 2,
    "evening": 2,
    "night": 2,
    "goodbye": 2,
    "casual": 1,

    "thanks": 4,
    "user-agree": 4,
    "user-advice": 4,
    "affirm": 4,
    "happy": 3,
    "jokes": 3,
    "friends": 3,

    "neutral-response": 0,
    "no-response": -1,
    "something-else": 0,
    "wrong": -1,
    "default": 0,
    "repeat": -0.5,

    "help": 3,
    "learn-more": 3,
    "about": 2,
    "skill": 3,
    "location": 2,
    "creation": 3,
    "ask": 3,
    "problem": 2,
    "no-approach": 0,
    "ask_about_mental_health": 3,
    "ask_coping": 3,

    "sad": -1,
    "stressed": -1,
    "worthless": -2,
    "anxious": -2,
    "overwhelmed": -2,
    "lonely": -2,
    "hopeless": -2,

    "depressed": -3,
    "scared": -3,
    "angry": -3,
    "worried": -2,
    "hurt": -3,
    "suicide": -5,
    "death": -4,
    "hate-you": -3,
    "hate-me": -3,
    "stupid": -3,

    "deny": -1,
    "done": 1,
    "not-talking": 1,
    "sleep": 1,
    "understand": 2,
    "user-meditation": 2,
    "meditation": 2,
    "pandora-useful": 1,
    "time_penalty": -0.1,
}

def load_domain(domain_path):
    with open(domain_path, "r") as f:
        d = yaml.safe_load(f)
    actions = d.get("actions", []) or []
    intents = d.get("intents", []) or []
    responses = d.get("responses", {}) if "responses" in d else d.get("templates", {})
    return intents, actions, responses

class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

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
            preds = policy_net(states)
            loss = nn.functional.nll_loss(torch.log(preds + 1e-9), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"BC epoch {ep+1}/{epochs} loss: {np.mean(losses):.4f}")

class EnhancedUserSimulator:
    def __init__(self, intents, max_turns=8):
        self.all_intents = intents
        self.max_turns = max_turns

        self.greetings = [
            "greeting", "morning", "afternoon", "evening", "night",
            "goodbye", "casual", "thanks", "jokes", "happy"
        ]

        self.positive = [
            "affirm", "user-agree", "user-advice", "understand",
            "user-meditation", "meditation", "pandora-useful"
        ]

        self.informational = [
            "about", "skill", "help", "learn-more", "creation",
            "location", "ask", "problem", "ask_about_mental_health",
            "ask_coping", "no-approach", "something-else", "friends"
        ]

        self.emotional = [
            "sad", "stressed", "worthless", "depressed", "anxious",
            "overwhelmed", "lonely", "hopeless", "worried"
        ]

        self.critical = [
            "suicide", "death", "hurt", "angry", "scared",
            "hate-you", "hate-me", "stupid"
        ]

        self.fallback = [
            "no-response", "wrong", "repeat", "default",
            "neutral-response"
        ]

        self.meta = [
            "deny", "done", "not-talking", "sleep"
        ]

        self.state = {}
        self.reset()

    def reset(self):
        self.turn = 0
        self.current_intent = random.choice(self.all_intents)
        self.emotion_depth = 0
        self.critical_handled = False
        return self.current_intent, f"[{self.current_intent}] initial"

    def respond(self, bot_action_idx, actions_list):
        self.turn += 1
        action_name = actions_list[bot_action_idx] if 0 <= bot_action_idx < len(actions_list) else "unknown"
        a_lower = action_name.lower()
        reward = REWARD_MAP.get("time_penalty", -0.1)
        if reward is None:
            reward = 0.0
        done = False
        user_text = f"[{self.current_intent}] continue"

        if self.current_intent in self.greetings:
            reward += REWARD_MAP.get("neutral", 1)
            if random.random() < 0.3:
                self.current_intent = random.choice(self.emotional)
            else:
                self.current_intent = random.choice(self.greetings + self.fallback)
            user_text = f"[{self.current_intent}] greeting-response"

        elif self.current_intent in self.positive:
            reward += REWARD_MAP.get("thanks", 3)
            done = True
            user_text = f"[{self.current_intent}] positive-end"

        elif self.current_intent in self.emotional:
            self.emotion_depth += 1
            if "help" in a_lower or "advice" in a_lower or "cope" in a_lower or "meditation" in a_lower:
                reward += REWARD_MAP.get("empathetic_reply", 2)
                if random.random() < 0.5:
                    reward += REWARD_MAP.get("thanks", 3)
                    self.current_intent = random.choice(self.greetings + self.positive)
                    done = True
                    user_text = f"[{self.current_intent}] thankful"
                else:
                    user_text = f"[{self.current_intent}] thinking"
            else:
                reward += REWARD_MAP.get("neutral", 1)
                user_text = f"[{self.current_intent}] needs-empathy"
            if self.emotion_depth >= 3:
                done = True

        elif self.current_intent in self.critical:
            if "help" in a_lower or "cope" in a_lower or "meditation" in a_lower:
                reward += REWARD_MAP.get("empathetic_reply", 2) + 2
                self.critical_handled = True
                done = True
                user_text = f"[{self.current_intent}] calmed"
            else:
                reward -= 3
                user_text = f"[{self.current_intent}] worsening"
                if self.turn >= self.max_turns:
                    done = True

        else:
            reward += REWARD_MAP.get("fallback", -2)
            user_text = f"[{self.current_intent}] fallback-response"

        if self.turn >= self.max_turns:
            done = True

        return self.current_intent, user_text, reward, done, {"action": action_name}

def reinforce_train(policy_net, simulator, actions_list, intent_to_idx, num_slots, episodes=500):
    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    avg_rewards = []

    for ep in range(episodes):
        intent, _ = simulator.reset()
        last_action_idx = 0
        done = False
        log_probs, rewards = [], []

        while not done:
            s_vec = build_state_vector(intent, {}, last_action_idx, intent_to_idx, num_slots, len(actions_list))
            s_tensor = torch.tensor(s_vec, device=DEVICE).unsqueeze(0)
            probs = policy_net(s_tensor)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            logp = dist.log_prob(a)
            action_idx = a.item()

            next_intent, _, r, done, _ = simulator.respond(action_idx, actions_list)

            if done and r > 0:
                r += 5

            log_probs.append(logp)
            rewards.append(r)
            last_action_idx = action_idx
            intent = next_intent

        if len(rewards) == 0:
            continue

        total_reward = sum(rewards) / len(rewards)

        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=DEVICE)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(f"[REINFORCE] Episode {ep+1}/{episodes}, avg_reward(last50): {np.mean(avg_rewards[-50:]):.3f}")

    return avg_rewards

def main():
    root = os.getcwd()
    model_path = os.path.join(root, MODEL_ARCHIVE)

    print(f"📦 Loading Rasa model from: {model_path}")
    agent = Agent.load(model_path)
    print("✅ Core agent loaded.")

    async def parse_fn(text):
        return await agent.parse_message(text)

    agent.parse_fn = parse_fn

    intents, actions, responses = load_domain(DOMAIN_FILE)
    print(f"Loaded {len(intents)} intents and {len(actions)} actions.")
    intent_to_idx = {it: i for i, it in enumerate(intents)}
    num_slots = 12

    bc_dataset, num_actions = asyncio.run(
        collect_bc_dataset(agent, intents, actions, intent_to_idx, num_slots=num_slots, examples_per_intent=6)
    )

    input_dim = len(intent_to_idx) + num_slots + num_actions
    policy = PolicyNet(input_dim, HIDDEN, num_actions).to(DEVICE)

    print("🚀 Starting behavior cloning...")
    behavior_cloning_train(policy, bc_dataset, epochs=BC_EPOCHS)

    print("🎯 Starting REINFORCE fine-tuning...")
    sim = EnhancedUserSimulator(intents)
    rewards = reinforce_train(policy, sim, actions, intent_to_idx, num_slots, episodes=RL_EPISODES)

    torch.save({
        "state_dict": policy.state_dict(),
        "intent_to_idx": intent_to_idx,
        "actions": actions,
        "num_slots": num_slots
    }, POLICY_SAVE)
    print("✅ Saved fine-tuned policy to", POLICY_SAVE)

    try:
        plt.figure(figsize=(8,4))
        window = max(1, len(rewards)//50)
        smoothed = [np.mean(rewards[max(0,i-window):i+1]) for i in range(len(rewards))]
        plt.plot(smoothed)
        plt.title("Smoothed episode reward (REINFORCE)")
        plt.xlabel("Episode")
        plt.ylabel("Total reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_curve.png")
        print("📊 Saved reward curve to reward_curve.png")
    except Exception as e:
        print("⚠️ Plotting failed:", e)


if __name__ == "__main__":
    main()
