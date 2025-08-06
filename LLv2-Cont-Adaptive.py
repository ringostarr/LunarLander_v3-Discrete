import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions.transforms import TanhTransform
from torch.distributions import TransformedDistribution
from transformers import pipeline
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ActorCriticNetwork(nn.Module):
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

        # Policy head
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim))

    def forward(self, state):
        shared_features = self.shared(state)
        value = self.value_head(shared_features)
        mu = self.mu_head(shared_features)
        std = torch.exp(self.log_std.clamp(-5, 20))  # Clamp for stability
        return mu, std, value

    def get_action_and_value(self, state):
        mu, std, value = self.forward(state)
        base_dist = Normal(mu, std)
        raw_action = base_dist.rsample()  # Sample from Normal
        action = torch.clamp(torch.tanh(raw_action),min=-0.99,max=0.99)  # Squash

        log_prob = base_dist.log_prob(raw_action)
        log_prob -= 2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))  # Tanh correction
        log_prob = log_prob.sum(dim=-1)
        entropy = base_dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value


class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.999, gae_lambda=0.98, value_coeff=0.5,
                 max_grad_norm=0.5, n_steps=2048, n_epochs=10, minibatch_size=128,
                 clip_range=0.2, entropy_coeff=0.01, max_entropy_coeff=0.05, min_entropy_coeff=0.001):

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.clip_range = clip_range
        self.entropy_coeff = entropy_coeff
        self.max_entropy_coeff = max_entropy_coeff
        self.min_entropy_coeff = min_entropy_coeff
        self.side_reward_ema = 0.0
        self.vertical_reward_ema = 0.0
        self.ema_alpha = 0.05

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.network = ActorCriticNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.episode_returns = []
        self.current_episode_return = 0
        self.total_timesteps = 0
        self.state, _ = self.env.reset()

    def collect_rollouts(self):
        states = []
        actions = []
        rewards = []
        values = []
        dones = []
        log_probs = []

        current_state = self.state
        _nstep = 0
        side_rewards = []
        vertical_rewards = []
        for step in range(self.n_steps):
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

            action_np = action.cpu().numpy().squeeze()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(current_state)
            actions.append(action.squeeze())
            rewards.append((reward/50))
            values.append(value.squeeze())
            dones.append(done)
            log_probs.append(log_prob.squeeze())
            pos_x, pos_y = current_state[0], current_state[1]
            vel_x, vel_y = current_state[2], current_state[3]
            angle = current_state[4]
            side_reward = -abs(pos_x) - abs(vel_x) - angle
            vertical_reward = -abs(pos_y) - abs(vel_y) + current_state[6]+current_state[7]

            if self.side_reward_ema == 0:
                self.side_reward_ema = side_reward
            else:
                self.side_reward_ema = self.ema_alpha * side_reward + (1 - self.ema_alpha) * self.side_reward_ema

            if self.vertical_reward_ema == 0:
                self.vertical_reward_ema = vertical_reward
            else:
                self.vertical_reward_ema = self.ema_alpha * vertical_reward + (1 - self.ema_alpha) * self.vertical_reward_ema
            side_rewards.append(self.side_reward_ema)
            vertical_rewards.append(self.vertical_reward_ema)
            self.current_episode_return += reward
            self.total_timesteps += 1

            if done:
                self.episode_returns.append(self.current_episode_return)
                self.current_episode_return = 0
                _nstep = 0
                current_state, _ = self.env.reset()
            else:
                current_state = next_state
                _nstep += 1

        self.state = current_state

        # Convert to tensors

        side_rewards_tensor = torch.FloatTensor(side_rewards)  # shape: [T]
        vertical_rewards_tensor = torch.FloatTensor(vertical_rewards)
        states_tensor = torch.stack([torch.FloatTensor(s) for s in states]).to(device)
        actions_tensor = torch.stack(actions).to(device)
        rewards_tensor = torch.FloatTensor(rewards).to(device)
        values_tensor = torch.stack(values).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(device)
        log_probs_tensor = torch.stack(log_probs).to(device)
        side_reward_ema = side_rewards_tensor.to(device)
        vertical_reward_ema = vertical_rewards_tensor.to(device)

        # Compute next value for GAE
        with torch.no_grad():
            if dones[-1]:
                next_value = torch.tensor(0.0).to(device)
            else:
                state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(device)
                _, _, next_value = self.network(state_tensor)
                next_value = next_value.squeeze()

        returns, advantages = self.compute_gae(rewards_tensor, values_tensor, dones_tensor, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            'states': states_tensor,
            'actions': actions_tensor,
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': log_probs_tensor,
            'old_values': values_tensor.clone().detach(),
            'side_reward_ema':side_reward_ema,
            'vertical_reward_ema' : vertical_reward_ema
        }

    def smooth_ema_per_sample(self, reward_tensor, alpha=0.1):
        if reward_tensor.numel() == 0:
            print("safety")
            return reward_tensor  # return empty tensor safely

        ema_rewards = torch.zeros_like(reward_tensor)
        ema_rewards[0] = reward_tensor[0]
        for t in range(1, reward_tensor.shape[0]):
            ema_rewards[t] = alpha * reward_tensor[t] + (1 - alpha) * ema_rewards[t - 1]
        return ema_rewards
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        gae = 0

        values_ext = torch.cat([values, next_value.unsqueeze(0)])

        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i].float()
            delta = rewards[i] + self.gamma * values_ext[i + 1] * mask - values_ext[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[i] = gae

        returns = advantages + values
        return returns, advantages

    def update(self, rollouts):
        states = rollouts['states']
        actions = rollouts['actions']
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        old_log_probs = rollouts['old_log_probs']

        batch_indices = np.arange(self.n_steps)
        policy_losses, value_losses, entropy_losses = [], [], []

        # Store entropy coeffs for logging
        all_entropy_coeffs = []

        for epoch in range(self.n_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, self.n_steps, self.minibatch_size):
                end = min(start + self.minibatch_size, self.n_steps)
                indices = batch_indices[start:end]

                mb_states = states[indices]
                mb_actions = actions[indices]
                mb_returns = returns[indices]
              #  mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
                mb_advantages = advantages[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_old_values = rollouts['old_values'][indices]
               # mb_old_values = (mb_old_values - mb_old_values.mean()) / (mb_old_values.std() + 1e-8)
                # Compute adaptive entropy coefficients based on states
                side_rewards = rollouts['side_reward_ema'][indices]
                vertical_rewards = rollouts['vertical_reward_ema'][indices]
                # Scale entropy coefficients
                vertical_entropy = self.scale_entropy_coeff(side_rewards)
                side_entropy_coeff = self.scale_entropy_coeff(vertical_rewards)
                # Create entropy coefficients tensor
               # side_entropy_coeff = torch.tensor(side_entropy_coeff, device=device)
                #vertical_entropy = torch.tensor(vertical_entropy, device=device)
                entropy_coeffs = torch.stack([side_entropy_coeff, vertical_entropy], dim=1).to(device)

                all_entropy_coeffs.append(entropy_coeffs)

                # Forward pass
                mu, std, values = self.network(mb_states)
                base_dist = Normal(mu, std)
                clipped_actions = torch.clamp(mb_actions, -0.999, 0.999)

                # Inverse tanh (atanh) to recover raw action before squashing
                raw_actions = 0.5 * (torch.log1p(clipped_actions) - torch.log1p(-clipped_actions))

                # Compute log probs under the current policy
                log_prob = base_dist.log_prob(raw_actions)
                log_prob -= 2 * (np.log(2) - raw_actions - F.softplus(-2 * raw_actions))  # Tanh correction
                log_prob = log_prob.sum(dim=-1)
                entropy_per_action = base_dist.entropy()  # Shape: [batch_size, action_dim]
                values = values.squeeze()
                #values = (values - values.mean()) / (values.std() + 1e-8)
                #old_values = rollouts['old_values']
                # PPO clipped loss
                ratio = torch.exp(log_prob - mb_old_log_probs.detach())
               # clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                #policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                epsilon = self.clip_range
                policy_loss_elements = ratio * mb_advantages - (mb_advantages.abs() / (2 * epsilon)) * (ratio - 1).pow(2)
                policy_loss = -policy_loss_elements.mean()

                # Value loss
                value_pred_clipped = mb_old_values + (values - mb_old_values).clamp(-0.2, 0.2)
                value_loss_clipped = (mb_returns - value_pred_clipped).pow(2)
                value_loss_unclipped = (mb_returns - values).pow(2)
                value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()

                # Adaptive entropy loss (action-dimension-wise)
                weighted_entropy = entropy_per_action * entropy_coeffs  # Element-wise multiplication
                entropy_loss = -weighted_entropy.sum(dim=-1).mean()  # Sum across actions, mean across batch

                # Total loss
                total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        # Compute average entropy coefficients for logging
        if all_entropy_coeffs:
            avg_entropy_coeffs = torch.stack(all_entropy_coeffs, dim=0)  # shape: [64, 64, 2]
            avg_entropy_coeffs = avg_entropy_coeffs.mean(dim=(0, 1))
            avg_side_entropy_coeff = avg_entropy_coeffs[0].item()
            avg_vertical_entropy_coeff = avg_entropy_coeffs[1].item()
        else:
            avg_side_entropy_coeff = 0.0
            avg_vertical_entropy_coeff = 0.0

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'avg_side_entropy_coeff': avg_side_entropy_coeff,
            'avg_vertical_entropy_coeff': avg_vertical_entropy_coeff
        }

    def scale_entropy_coeff(self, reward, min_reward=-2, max_reward=0, min_coeff=0.01, max_coeff=0.01):
        reward_tensor = torch.tensor(reward) if not isinstance(reward, torch.Tensor) else reward
        reward_clipped = torch.clamp(reward_tensor, min=min_reward, max=max_reward)
        t = (reward_clipped - min_reward) / (max_reward - min_reward)  # scales to [0, 1]
        return max_coeff - t * (max_coeff - min_coeff)

    def train(self, total_timesteps=1_000_000):
        num_updates = total_timesteps // self.n_steps
        print(f"Starting training for {num_updates} updates ({total_timesteps} timesteps)")

        for update in range(num_updates):
            rollouts = self.collect_rollouts()
            metrics = self.update(rollouts)

            if update % 10 == 0:
                avg_return = (np.mean(self.episode_returns[-50:])
                              if len(self.episode_returns) >= 50
                              else np.mean(self.episode_returns) if self.episode_returns else 0)
                print(f"Update {update:4d}, Timesteps: {self.total_timesteps:7d}, "
                      f"Episodes: {len(self.episode_returns):4d}, Avg Return: {avg_return:7.2f}")
                print(f"  Policy Loss: {metrics['policy_loss']:7.4f}, "
                      f"Value Loss: {metrics['value_loss']:7.4f}, "
                      f"Entropy Loss: {metrics['entropy_loss']:7.4f}")
                print(f"  Avg Side Entropy Coeff: {metrics['avg_side_entropy_coeff']:.4f}, "
                      f"Avg Vertical Entropy Coeff: {metrics['avg_vertical_entropy_coeff']:.4f}")
                print("-" * 80)

    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_returns': self.episode_returns,
            'total_timesteps': self.total_timesteps
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_returns = checkpoint['episode_returns']
        self.total_timesteps = checkpoint['total_timesteps']
        print(f"Model loaded from {path}")


def plot_training_progress(episode_returns, window=50):
    if len(episode_returns) == 0:
        print("No episodes completed to plot.")
        return

    episode_returns = np.array(episode_returns)
    if len(episode_returns) < window:
        window = len(episode_returns)

    moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(episode_returns)), episode_returns, alpha=0.3,
             label='Episode Returns', color='lightblue')
    plt.plot(range(window - 1, len(episode_returns)), moving_avg,
             color='red', linewidth=2, label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('PPO Training Progress - LunarLander Continuous')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test_agent(agent, num_episodes=5):
    test_returns = []

    for episode in range(num_episodes):
        state, _ = agent.env.reset()
        episode_return = 0
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _, _ = agent.network.get_action_and_value(state_tensor)

            action_np = action.cpu().numpy().squeeze()
            state, reward, terminated, truncated, _ = agent.env.step(action_np)
            done = terminated or truncated
            episode_return += reward

        test_returns.append(episode_return)
        print(f"Test Episode {episode + 1}: Return = {episode_return:.2f}")

    avg_return = np.mean(test_returns)
    print(f"Average Test Return: {avg_return:.2f} ± {np.std(test_returns):.2f}")
    return test_returns


if __name__ == "__main__":
    # Create environment
    env = gym.make('LunarLanderContinuous-v3')


    # Create PPO agent with adaptive entropy
    agent = PPOAgent(
        env=env,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.98,
        n_steps=2048,
        n_epochs=10,
        minibatch_size=256,
        clip_range=0.2,
        entropy_coeff=0.01,  # Base entropy coefficient (not used in adaptive version)
        max_entropy_coeff=0.02,
        value_coeff=0.6,# Maximum entropy coefficient for scaling
        min_entropy_coeff=0.005  # Minimum entropy coefficient for scaling
    )

    try:

        # Train the agent
        agent.train(total_timesteps=1_000_000)

        # Save the model
        agent.save_model('ppo_lunarlander.pth')

        # Plot training progress
        plot_training_progress(agent.episode_returns)

        # Test the trained agent
        print("\nTesting trained agent...")
        test_agent(agent, num_episodes=50)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if agent.episode_returns:
            plot_training_progress(agent.episode_returns)

    finally:
        env.close()