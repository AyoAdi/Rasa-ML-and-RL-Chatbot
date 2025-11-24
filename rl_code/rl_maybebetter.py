# rl_train_enhanced.py — Enhanced REINFORCE fine-tuning for Rasa TEDPolicy
"""
Enhanced RL fine-tuning (Behavior Cloning + improved REINFORCE).
Run from project root:
  python3 rl_train_enhanced.py
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
import pandas as pd
from collections import namedtuple, deque

# Rasa imports
from rasa.core.agent import Agent

# ----------------------------
# Config (tuned)
# ----------------------------
MODEL_ARCHIVE = "models/20251106-152304-tempered-louver.tar.gz"
DOMAIN_FILE = "domain.yml"

BC_EPOCHS = 25
RL_EPISODES = 1200              # increase episodes for better convergence
GAMMA = 0.99
LR = 3e-4                       # slightly higher LR for faster learning with capacity
HIDDEN = 256                    # larger hidden size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POLICY_SAVE = "policy_finetuned_enhanced.pt"
SEED = 42

# Reward clipping to avoid outlier explosion
REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 5.0

# Stability / exploration hyperparams
ENTROPY_COEF = 0.01             # entropy bonus coefficient
MAX_GRAD_NORM = 0.5
UPDATE_EVERY = 8                # accumulate gradients over N episodes
BASELINE_WINDOW = 200           # moving-window baseline size
SCHED_STEP = 300                # scheduler step size
SCHED_GAMMA = 0.8               # LR decay factor

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Reward map (kept, but you can tune these)
# ----------------------------
REWARD_MAP = {
    "greeting": 2, "morning": 2, "afternoon": 2, "evening": 2, "night": 2,
    "goodbye": 2, "casual": 1, "thanks": 4, "user-agree": 4, "user-advice": 4,
    "affirm": 4, "happy": 3, "jokes": 3, "friends": 3, "neutral-response": 0,
    "no-response": -1, "something-else": 0, "wrong": -1, "default": 0,
    "repeat": -0.5, "help": 3, "learn-more": 3, "about": 2, "skill": 3,
    "location": 2, "creation": 3, "ask": 3, "problem": 2, "no-approach": 0,
    "ask_about_mental_health": 3, "ask_coping": 3,
    # emotional states: slightly reduce extreme negatives so learning isn't dominated by rare huge negatives
    "sad": -1, "stressed": -1, "worthless": -2, "anxious": -2, "overwhelmed": -2, "lonely": -2,
    "hopeless": -2, "depressed": -3, "scared": -3, "angry": -3, "worried": -2,
    "hurt": -3, "suicide": -4, "death": -3, "hate-you": -3, "hate-me": -3, "stupid": -3,
    "deny": -1, "done": 1, "not-talking": 1, "sleep": 1, "understand": 2,
    "user-meditation": 2, "meditation": 2, "pandora-useful": 1, "time_penalty": -0.05
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
# Simple policy network (same API)
# ----------------------------
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

# ----------------------------
# State builder (same)
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
# BC dataset collection (same)
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
# Behavior cloning (same)
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
            preds = policy_net(states)
            loss = nn.functional.nll_loss(torch.log(preds + 1e-9), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"BC epoch {ep+1}/{epochs} loss: {np.mean(losses):.4f}")

# ----------------------------
# Enhanced User Simulator (same but with clipped rewards)
# ----------------------------
class EnhancedUserSimulator:
    def __init__(self, intents, max_turns=8):
        self.all_intents = intents
        self.max_turns = max_turns
        # categories
        self.greetings = ["greeting", "morning", "afternoon", "evening", "night", "goodbye", "casual", "thanks", "jokes", "happy"]
        self.positive = ["affirm", "user-agree", "user-advice", "understand", "user-meditation", "meditation", "pandora-useful"]
        self.informational = ["about", "skill", "help", "learn-more", "creation", "location", "ask", "problem", "ask_about_mental_health", "ask_coping", "no-approach", "something-else", "friends"]
        self.emotional = ["sad", "stressed", "worthless", "depressed", "anxious", "overwhelmed", "lonely", "hopeless", "worried"]
        self.critical = ["suicide", "death", "hurt", "angry", "scared", "hate-you", "hate-me", "stupid"]
        self.fallback = ["no-response", "wrong", "repeat", "default", "neutral-response"]
        self.meta = ["deny", "done", "not-talking", "sleep"]
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
        reward = REWARD_MAP.get("time_penalty", -0.05)
        done = False
        user_text = f"[{self.current_intent}] continue"

        if self.current_intent in self.greetings:
            reward += 1
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
                reward += 2
                if random.random() < 0.5:
                    # successful empathetic handling
                    reward += 3
                    self.current_intent = random.choice(self.greetings + self.positive)
                    done = True
                    user_text = f"[{self.current_intent}] thankful"
                else:
                    user_text = f"[{self.current_intent}] thinking"
            else:
                reward += 1
                user_text = f"[{self.current_intent}] needs-empathy"
            if self.emotion_depth >= 3:
                done = True

        elif self.current_intent in self.critical:
            if "help" in a_lower or "cope" in a_lower or "meditation" in a_lower:
                reward += 4
                self.critical_handled = True
                done = True
                user_text = f"[{self.current_intent}] calmed"
            else:
                reward -= 3
                user_text = f"[{self.current_intent}] worsening"
                if self.turn >= self.max_turns:
                    done = True

        else:
            reward -= 2
            user_text = f"[{self.current_intent}] fallback-response"

        if self.turn >= self.max_turns:
            done = True

        # clip reward to safe range
        reward = float(np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))
        return self.current_intent, user_text, reward, done, {"action": action_name}

# ----------------------------
# REINFORCE training (improvements: reward-to-go, baseline, entropy, batch updates)
# ----------------------------
def reinforce_train(policy_net, simulator, actions_list, intent_to_idx, num_slots, episodes=500):
    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHED_STEP, gamma=SCHED_GAMMA)

    episode_mean_rewards = []
    episode_total_rewards = []

    # moving-window baseline (for advantage)
    baseline_window = deque(maxlen=BASELINE_WINDOW)

    accumulated_steps = 0

    for ep in range(episodes):
        intent, _ = simulator.reset()
        last_action_idx = 0
        done = False

        log_probs = []
        rewards = []
        entropies = []

        # collect one episode
        while not done:
            s_vec = build_state_vector(intent, {}, last_action_idx, intent_to_idx, num_slots, len(actions_list))
            s_tensor = torch.tensor(s_vec, device=DEVICE).unsqueeze(0)
            probs = policy_net(s_tensor)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            logp = dist.log_prob(a)
            ent = dist.entropy()
            action_idx = a.item()

            next_intent, _, r, done, _ = simulator.respond(action_idx, actions_list)

            # terminal positive bonus (clipped)
            if done and r > 0:
                r = float(np.clip(r + 3.0, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

            log_probs.append(logp)
            entropies.append(ent)
            rewards.append(float(np.clip(r, REWARD_CLIP_MIN, REWARD_CLIP_MAX)))

            last_action_idx = action_idx
            intent = next_intent

        # if empty episode skip
        if len(rewards) == 0:
            continue

        # compute reward-to-go returns (G_t) for each timestep t in episode
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32)

        # normalize returns (optional) - keep scale sensible
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) if len(returns) > 1 else returns

        # compute baseline from window as scalar; baseline broadcasted for whole episode
        mean_per_step = float(sum(rewards) / len(rewards))
        baseline_window.append(mean_per_step)
        baseline = float(np.mean(baseline_window)) if len(baseline_window) > 0 else 0.0
        baseline_tensor = torch.full_like(returns, baseline, device=DEVICE, dtype=torch.float32)

        # advantage = returns - baseline
        advantages = returns - baseline_tensor

        # normalize advantages (helps further reduce variance)
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # compute policy loss (REINFORCE with advantage)
        policy_loss_terms = [-lp * adv for lp, adv in zip(log_probs, advantages)]
        policy_loss = torch.stack(policy_loss_terms).sum()

        # entropy regularization (encourage exploration)
        ent_loss = -ENTROPY_COEF * torch.stack(entropies).sum()

        loss = policy_loss + ent_loss

        # accumulate gradients (scale by UPDATE_EVERY so each final update is average of N episodes)
        (loss / UPDATE_EVERY).backward()
        accumulated_steps += 1

        # store stats
        total_reward = float(sum(rewards))
        mean_per_timestep = float(total_reward / len(rewards))
        episode_total_rewards.append(total_reward)
        episode_mean_rewards.append(mean_per_timestep)

        # perform optimizer step when enough episodes accumulated
        if accumulated_steps >= UPDATE_EVERY:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulated_steps = 0

        # periodic logging
        if (ep + 1) % 50 == 0:
            recent_mean = np.mean(episode_mean_rewards[-50:]) if len(episode_mean_rewards) >= 50 else np.mean(episode_mean_rewards)
            print(f"[REINFORCE+] Ep {ep+1}/{episodes} mean(last50)={recent_mean:.3f} baseline={baseline:.3f} lr={scheduler.get_last_lr()[0]:.2e}")

    # final step if leftover gradients
    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return episode_mean_rewards, episode_total_rewards

# ----------------------------
# MAIN
# ----------------------------
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

    # RL fine-tune
    print("🎯 Starting enhanced REINFORCE fine-tuning...")
    sim = EnhancedUserSimulator(intents)
    mean_rewards, total_rewards = reinforce_train(policy, sim, actions, intent_to_idx, num_slots, episodes=RL_EPISODES)

    # Save fine-tuned policy
    torch.save({
        "state_dict": policy.state_dict(),
        "intent_to_idx": intent_to_idx,
        "actions": actions,
        "num_slots": num_slots
    }, POLICY_SAVE)
    print("✅ Saved fine-tuned policy to", POLICY_SAVE)

    # ----------------------------
    # Stats & plotting
    # ----------------------------
    rewards_array = np.array(mean_rewards)
    total_array = np.array(total_rewards)

    mean_reward = float(np.mean(rewards_array))
    median_reward = float(np.median(rewards_array))
    std_reward = float(np.std(rewards_array))
    var_reward = float(np.var(rewards_array))
    min_reward = float(np.min(rewards_array))
    max_reward = float(np.max(rewards_array))
    last50_mean = float(np.mean(rewards_array[-50:])) if len(rewards_array) >= 50 else mean_reward
    p25 = float(np.percentile(rewards_array, 25))
    p75 = float(np.percentile(rewards_array, 75))
    p90 = float(np.percentile(rewards_array, 90))

    print("\n📈 Reinforcement Learning Summary (mean-per-step rewards)")
    print("-----------------------------------")
    print(f"Total episodes               : {len(rewards_array)}")
    print(f"Mean reward                 : {mean_reward:.3f}")
    print(f"Median reward               : {median_reward:.3f}")
    print(f"Std. deviation              : {std_reward:.3f}")
    print(f"Variance                    : {var_reward:.3f}")
    print(f"Min reward                  : {min_reward:.3f}")
    print(f"Max reward                  : {max_reward:.3f}")
    print(f"25th percentile             : {p25:.3f}")
    print(f"75th percentile             : {p75:.3f}")
    print(f"90th percentile             : {p90:.3f}")
    print(f"Mean of last 50 episodes    : {last50_mean:.3f}")
    print("-----------------------------------\n")

    stats_df = pd.DataFrame({
        "metric": ["episodes", "mean", "median", "std_dev", "variance", "min", "max", "25pct", "75pct", "90pct", "last50_mean"],
        "value": [len(rewards_array), mean_reward, median_reward, std_reward, var_reward, min_reward, max_reward, p25, p75, p90, last50_mean]
    })
    stats_df.to_csv("rl_training_stats_enhanced.csv", index=False)
    print("✅ Saved detailed statistics to rl_training_stats_enhanced.csv")

    # Plot smoothed curve
    try:
        plt.figure(figsize=(10, 4.8))
        window = max(1, len(rewards_array) // 50)
        smoothed = [np.mean(rewards_array[max(0, i - window):i + 1]) for i in range(len(rewards_array))]
        plt.plot(smoothed, label="Smoothed Reward", color="blue", linewidth=1.3)
        plt.axhline(mean_reward, color="green", linestyle="--", label=f"Mean = {mean_reward:.2f}")
        lower = [np.percentile(rewards_array[max(0, i - window):i + 1], 10) for i in range(len(rewards_array))]
        upper = [np.percentile(rewards_array[max(0, i - window):i + 1], 90) for i in range(len(rewards_array))]
        plt.fill_between(range(len(smoothed)), np.array(lower), np.array(upper), color="lightblue", alpha=0.25, label="10–90 percentile")
        plt.title("Smoothed Episode Reward (Enhanced REINFORCE)")
        plt.xlabel("Episode")
        plt.ylabel("Mean per-step Reward")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_curve_enhanced.png", dpi=150)
        print("📊 Saved detailed reward curve to reward_curve_enhanced.png")
    except Exception as e:
        print("⚠️ Detailed plotting failed:", e)

    # Save per-episode rewards for reproducibility
    pd.DataFrame({"episode_mean_reward": rewards_array, "episode_total_reward": total_array}).to_csv("rl_episode_rewards_enhanced.csv", index_label="episode")
    print("✅ Saved per-episode rewards to rl_episode_rewards_enhanced.csv")

if __name__ == "__main__":
    main()
