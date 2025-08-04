import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt


# Define the neural network architecture
class ActorCriticNetwork(nn.Module):
    """
    A shared neural network for both the policy (actor) and value function (critic).
    This architecture is common for both A2C and PPO.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared layers process the state input
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # The critic's head predicts the state value
        self.value_head = nn.Linear(hidden_dim, 1)

        # The actor's head predicts the logits for each action
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Performs a forward pass to get the value and action logits.
        """
        shared_features = self.shared(state)
        value = self.value_head(shared_features)
        logits = self.policy_head(shared_features)
        return logits, value

    def get_action_and_value(self, state):
        """
        Takes a state and returns an action, its log probability,
        the policy's entropy, and the estimated state value.
        """
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value


class PPOAgent:
    """
    A PPO agent configured to train on LunarLander-v3.
    """

    def __init__(self, env, lr=3e-4, gamma=0.999, gae_lambda=0.95, value_coeff=0.5,
                 max_grad_norm=0.5, n_steps=2048, n_epochs=10, minibatch_size=64,
                 clip_range=0.2,
                 ema_alpha=0.1):
        """
        Initializes the PPO agent with specific hyperparameters.
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.clip_range = clip_range

        # Adaptive entropy parameters using a dual exponential decay/growth function
        self.ema_alpha = ema_alpha
        self.entropy_coeff = 0.01  # Initial entropy coefficient

        # Exponential decay constants for positive rewards
        # Coeff goes from 0.01 to 0.001 as reward goes from 0 to 200
        self.exp_decay_A_pos = 0.01
        self.exp_decay_B_pos = 0.01151

        # Exponential growth constants for negative rewards
        # Coeff goes from 0.01 to 0.03 as reward goes from 0 to -200 (approx)
        self.exp_decay_A_neg = 0.01
        self.exp_decay_B_neg = 0.00549

        # To make it more intuitive, let's define the min, mid, and max coeffs
        self.min_entropy_coeff = 0.001
        self.mid_entropy_coeff = 0.01
        self.max_entropy_coeff = 0.03

        # EMA tracking
        self.ema_reward = None

        # Network setup for LunarLander (discrete action space)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Tracking variables
        self.episode_returns = []
        self.current_episode_return = 0
        self.total_timesteps = 0
        self.state, _ = self.env.reset()
        self.episode_count = 0

    def get_current_state(self, is_test=False):
        """
        Handles environment resets and returns the current state.
        """
        if self.state is None or is_test:
            self.state, _ = self.env.reset()
        return self.state

    def collect_rollouts(self):
        """
        Collect a fixed number of transitions (n_steps) for a single PPO update.
        This function now stores old log probabilities for the PPO objective.
        """
        states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []

        current_state = self.get_current_state()

        for _ in range(self.n_steps):
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

            action_np = action.item()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(current_state)
            actions.append(action.squeeze())
            rewards.append(reward)
            values.append(value.squeeze())
            dones.append(done)
            log_probs.append(log_prob.squeeze())

            self.current_episode_return += reward
            self.total_timesteps += 1

            if done:
                # Episode finished, reset the environment and store the return
                self.episode_returns.append(self.current_episode_return)
                self.current_episode_return = 0
                current_state, _ = self.env.reset()
                self.episode_count += 1
            else:
                current_state = next_state

        self.state = current_state

        # Convert lists to tensors
        states_tensor = torch.stack([torch.FloatTensor(s) for s in states])
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        values_tensor = torch.stack(values)
        dones_tensor = torch.tensor(dones, dtype=torch.bool)
        log_probs_tensor = torch.stack(log_probs)

        # Compute GAE for advantages and returns
        with torch.no_grad():
            _, next_value = self.network(torch.FloatTensor(self.state).unsqueeze(0))
            next_value = next_value.squeeze()

        returns, advantages = self.compute_gae(rewards_tensor, values_tensor, dones_tensor, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Return a dictionary of the collected data
        return {
            'states': states_tensor,
            'actions': actions_tensor,
            'returns': returns,
            'advantages': advantages,
            'old_log_probs': log_probs_tensor
        }

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE) for the rollouts.
        """
        advantages = []
        advantage = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1.0 - dones[i].float()) - values[i]
            advantage = delta + self.gamma * self.gae_lambda * (1.0 - dones[i].float()) * advantage
            advantages.insert(0, advantage)
            next_value = values[i]

        returns = torch.FloatTensor(advantages) + values
        advantages = torch.FloatTensor(advantages)
        return returns, advantages

    def update2(self, rollouts):
        """
        Performs a PPO update step with an adaptive entropy coefficient.
        """
        # --- Update Adaptive Entropy Coefficient ---
        if self.episode_count > 0:
            last_returns = self.episode_returns[-self.episode_count:]
            avg_last_return = np.mean(last_returns)
            if self.ema_reward is None:
                self.ema_reward = avg_last_return
            else:
                self.ema_reward = (1 - self.ema_alpha) * self.ema_reward + self.ema_alpha * avg_last_return

            self.episode_count = 0

        if self.ema_reward is not None:
            if self.ema_reward > 0:
                # Exponential decay for positive rewards
                new_coeff = self.exp_decay_A_pos * np.exp(-self.exp_decay_B_pos * self.ema_reward)
                self.entropy_coeff = np.clip(new_coeff, self.min_entropy_coeff, self.mid_entropy_coeff)
            else:
                # Exponential growth for negative or zero rewards
                new_coeff = self.exp_decay_A_neg * np.exp(-self.exp_decay_B_neg * self.ema_reward)
                self.entropy_coeff = np.clip(new_coeff, self.mid_entropy_coeff, self.max_entropy_coeff)

        # Unpack rollouts
        states = rollouts['states']
        actions = rollouts['actions']
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        old_log_probs = rollouts['old_log_probs']

        # Get the number of minibatches
        num_minibatches = self.n_steps // self.minibatch_size
        batch_indices = np.arange(self.n_steps)

        for _ in range(self.n_epochs):
            # Shuffle indices to create different minibatches each epoch
            np.random.shuffle(batch_indices)

            for start in range(0, self.n_steps, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                # Get minibatch data
                minibatch_states = states[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_returns = returns[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                minibatch_old_log_probs = old_log_probs[minibatch_indices]

                # Forward pass for the minibatch
                logits, current_values = self.network(minibatch_states)
                dist = torch.distributions.Categorical(logits=logits)
                current_log_probs = dist.log_prob(minibatch_actions)
                current_entropies = dist.entropy()
                current_values = current_values.squeeze()

                # --- PPO Loss Calculation ---
                # Policy Ratio
                ratio = torch.exp(current_log_probs - minibatch_old_log_probs.detach())

                # Clipped Policy Loss
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(ratio * minibatch_advantages, clipped_ratio * minibatch_advantages).mean()

                # Value Loss
                value_loss = F.mse_loss(current_values, minibatch_returns.detach())

                # Entropy Loss (using the adaptive coefficient)
                entropy_loss = -self.entropy_coeff * current_entropies.mean()

                total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss

                # Update network parameters
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

    def train(self, total_timesteps=1_000_000):
        """
        Main training loop for the PPO agent.
        """
        num_updates = total_timesteps // self.n_steps
        for update in range(num_updates):
            rollouts = self.collect_rollouts()
            # Use update2 for adaptive entropy
            metrics = self.update2(rollouts)

            if update % 10 == 0:
                # Use a sliding window for printing average return
                if len(self.episode_returns) >= 50:
                    avg_return = np.mean(self.episode_returns[-50:])
                else:
                    avg_return = np.mean(self.episode_returns) if self.episode_returns else 0

                print(f"Update {update}, Timesteps: {self.total_timesteps}, Episodes: {len(self.episode_returns)}, "
                      f"Avg Return: {avg_return:.2f}, Entropy Coeff: {metrics['entropy_coeff']:.4f}, EMA rew: {metrics['ema_reward']:.4f}, "
                      f"Policy Loss: {metrics['policy_loss']:.4f}")


def plot_average_returns(episode_returns, window=50):
    """
    Plot the raw episode returns and a moving average to visualize training progress.
    """
    if len(episode_returns) == 0:
        print("No episodes completed to plot.")
        return

    episode_returns = np.array(episode_returns)
    moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode='valid')

    plt.figure(figsize=(12, 7))
    plt.plot(range(len(episode_returns)), episode_returns, alpha=0.3, label='Episode Returns')
    plt.plot(range(window - 1, len(episode_returns)), moving_avg, color='red', label=f'{window}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Progress: Episode Returns')
    plt.grid(True)
    plt.legend()
    plt.show()


# Example usage for LunarLander
if __name__ == "__main__":
    env = gym.make('LunarLander-v3')

    print("Training PPO Agent on LunarLander-v3")
    # Set entropy_power to a higher value for a more aggressive, non-linear decay
    agent = PPOAgent(env)

    agent.train(total_timesteps=600_000)

    print("Plotting results...")
    plot_average_returns(agent.episode_returns, 50)

    env.close()
