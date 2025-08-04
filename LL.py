import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
    """Shared network for both policy and value estimation"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Policy head (discrete actions for LunarLander)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        shared_features = self.shared(state)
        value = self.value_head(shared_features)
        logits = self.policy_head(shared_features)
        return logits, value

    def get_action_and_value(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class AdaptiveEntropyA2C:
    """A2C with EMA-based adaptive entropy coefficient for LunarLander"""

    def __init__(self, env, lr=3e-4, gamma=0.99, value_coeff=0.5,
                 max_grad_norm=0.5, n_steps=10,
                 # Entropy adaptation parameters
                 initial_entropy_coeff=0.01,
                 min_entropy_coeff=0.001,
                 max_entropy_coeff=0.1,
                 ema_alpha=0.15,
                 adaptation_rate=0.02,
                 reward_scale=100.0):

        self.env = env
        self.gamma = gamma
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        # Entropy adaptation parameters
        self.entropy_coeff = initial_entropy_coeff
        self.min_entropy_coeff = min_entropy_coeff
        self.max_entropy_coeff = max_entropy_coeff
        self.ema_alpha = ema_alpha
        self.adaptation_rate = adaptation_rate
        self.reward_scale = reward_scale

        # EMA tracking
        self.ema_reward = None
        self.episode_count = 0

        # Network setup for LunarLander (discrete action space)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Tracking
        self.episode_returns = []
        self.current_episode_return = 0

    def collect_rollouts(self, min_episodes=5):
        """Collect transitions until a minimum number of episodes are completed"""
        states, actions, rewards, log_probs, values, entropies, dones = [], [], [], [], [], [], []

        state, _ = self.env.reset()

        episodes_collected = 0
        steps = 0

        while episodes_collected < min_episodes:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, entropy, value = self.network.get_action_and_value(state_tensor)

            action_np = action.item()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action.squeeze())
            rewards.append(reward-0.001*steps)
            log_probs.append(log_prob.squeeze())
            values.append(value.squeeze())
            entropies.append(entropy.squeeze())
            dones.append(done)

            self.current_episode_return += reward
            steps += 1

            if done:
                self.episode_returns.append(self.current_episode_return)
                self.current_episode_return = 0
                state, _ = self.env.reset()
                episodes_collected += 1
                self.n_steps += steps
                steps = 0
            else:
                state = next_state

        return (torch.stack([torch.FloatTensor(s) for s in states]),
                torch.stack(actions),
                torch.FloatTensor(rewards),
                torch.stack(log_probs),
                torch.stack(values),
                torch.stack(entropies),
                torch.tensor(dones, dtype=torch.bool))

    def collect_rollouts2(self):
        """Collect n-step rollouts for A2C"""
        states, actions, rewards, log_probs, values, entropies, dones = [], [], [], [], [], [], []

        state, _ = self.env.reset()

        for _ in range(self.n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, entropy, value = self.network.get_action_and_value(state_tensor)

            action_np = action.item()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action.squeeze())
            rewards.append(reward)
            log_probs.append(log_prob.squeeze())
            values.append(value.squeeze())
            entropies.append(entropy.squeeze())
            dones.append(done)

            self.current_episode_return += reward

            if done:
                # Episode finished
                self.episode_returns.append(self.current_episode_return)
                self.current_episode_return = 0
                state, _ = self.env.reset()
            else:
                state = next_state

        return (torch.stack([torch.FloatTensor(s) for s in states]),
                torch.stack(actions),
                torch.FloatTensor(rewards),
                torch.stack(log_probs),
                torch.stack(values),
                torch.stack(entropies),
                torch.tensor(dones, dtype=torch.bool))

    def compute_gae(self, rewards, values, dones, next_value=0, gae_lambda=0.98):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        advantage = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i].float()
                nextvalues = next_value
            else:
                next_non_terminal = 1.0 - dones[i].float()
                nextvalues = values[i + 1]

            delta = rewards[i] + self.gamma * nextvalues * next_non_terminal - values[i]
            advantage = delta + self.gamma * gae_lambda * next_non_terminal * advantage
            advantages.insert(0, advantage)

        returns = torch.FloatTensor(advantages) + values
        advantages = torch.FloatTensor(advantages)

        return returns, advantages

    def update(self):
        """A2C update step with adaptive entropy"""
        states, actions, rewards, log_probs, values, entropies, dones = self.collect_rollouts(min_episodes=5)

        # Update EMA with latest episode returns
        new_episodes = self.episode_returns[self.episode_count:]
        for episode_return in new_episodes:
            self.episode_count += 1
            if self.ema_reward is None:
                self.ema_reward = episode_return
            else:
                self.ema_reward = (1 - self.ema_alpha) * self.ema_reward + self.ema_alpha * episode_return

        # Adapt entropy coefficient based on EMA reward
        if self.episode_count >= 5 and self.ema_reward is not None:
            # Normalize reward to [0, 1] range
            reward_normalized = np.clip((self.ema_reward)/200, -1, 1)
            reward_normalized = (reward_normalized+1)/2

            # Map normalized reward to entropy coefficient
            # Low rewards (0) -> High entropy (more exploration)
            # High rewards (+1) -> Low entropy (more exploitation)
            target_entropy = self.min_entropy_coeff + (self.max_entropy_coeff - self.min_entropy_coeff) * (
                        1 - reward_normalized)

            # Smoothly adapt entropy coefficient
            #self.entropy_coeff += self.adaptation_rate * (target_entropy - self.entropy_coeff)
            self.entropy_coeff = target_entropy
            self.entropy_coeff = np.clip(self.entropy_coeff, self.min_entropy_coeff, self.max_entropy_coeff)

        # Get bootstrap value for last state
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(states[-1]).unsqueeze(0))
            next_value = next_value.squeeze()

        returns, advantages = self.compute_gae(rewards, values, dones, next_value)

        # Forward pass to get current policy outputs
        logits, current_values = self.network(states)
        dist = torch.distributions.Categorical(logits=logits)
        current_log_probs = dist.log_prob(actions)
        current_entropies = dist.entropy()
        current_values = current_values.squeeze()

        # Compute losses with current entropy coefficient
        policy_loss = -(current_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(current_values, returns)
        entropy_loss = -self.entropy_coeff * current_entropies.mean()

        total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'entropy_coeff': self.entropy_coeff,
            'ema_reward': self.ema_reward if self.ema_reward else 0
        }

    def train(self, num_updates=1000):
        """Training loop for A2C"""
        for update in range(num_updates):
           # metrics = self.update()
            metrics = self.update()

            if update % 100 == 0 and len(self.episode_returns) > 0:
                avg_return = np.mean(self.episode_returns[-50:])
                print(f"Update {update}, Episodes: {len(self.episode_returns)}, "
                      f"Avg Return: {avg_return:.2f}, EMA: {metrics['ema_reward']:.2f}, "
                      f"Entropy Coeff: {metrics['entropy_coeff']:.4f}, Entropy: {metrics['entropy_loss']:.4f}, policy Loss: {metrics['policy_loss']:.4f} ,"
                      f"n_timesteps: {self.n_steps}")


def plot_training_progress(agent):
    """Plot episode returns only"""
    if len(agent.episode_returns) > 0:
        episodes = range(len(agent.episode_returns))
        plt.plot(episodes, agent.episode_returns, alpha=0.6, label='Episode Returns')
        plt.title('Episode Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.grid(True)
        plt.show()
        print()


def plot_average_returns(episode_returns, window=50):
    episode_returns = np.array(episode_returns)

    # Compute moving average with 'valid' mode to avoid edge effects
    moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(episode_returns)), episode_returns, alpha=0.3, label='Episode Returns')
    plt.plot(range(window - 1, len(episode_returns)), moving_avg, color='red', label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Episode Returns and Moving Average')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage for LunarLander
if __name__ == "__main__":
    env = gym.make('LunarLander-v3')

    print("Training A2C on LunarLander-v3 with EMA-reward based adaptive entropy")
    agent = AdaptiveEntropyA2C(env,n_steps=50,
                               initial_entropy_coeff=0.01,
                               min_entropy_coeff=0.001,
                               max_entropy_coeff=0.01,
                               ema_alpha=0.1,
                               adaptation_rate=0.02,
                               reward_scale=200)

    agent.train(num_updates=10000)

    print("Plotting results...")
    #plot_training_progress(agent)
    plot_average_returns(agent.episode_returns,50)

    env.close()