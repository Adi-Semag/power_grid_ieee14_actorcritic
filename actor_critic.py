import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Voltage control branch
        self.voltage_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim // 3),
            nn.Tanh()
        )
        
        # Active power control branch
        self.active_power_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim // 3),
            nn.Tanh()
        )
        
        # Reactive power control branch
        self.reactive_power_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim // 3),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state, training=False):
        features = self.feature_net(state)
        
        # Apply dropout only during training
        if training:
            features = F.dropout(features, p=0.1)
        
        # Get control signals for each component
        voltage_control = self.voltage_net(features)
        active_power = self.active_power_net(features)
        reactive_power = self.reactive_power_net(features)
        
        # Combine control signals
        action = torch.cat([voltage_control, active_power, reactive_power], dim=-1)
        
        return action

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        
        # State feature extraction
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Action feature extraction
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
            module.bias.data.zero_()
    
    def forward(self, state, action, training=False):
        # Extract features
        state_features = self.state_net(state)
        action_features = self.action_net(action)
        
        # Apply dropout only during training
        if training:
            state_features = F.dropout(state_features, p=0.1)
            action_features = F.dropout(action_features, p=0.1)
        
        # Combine features
        combined = torch.cat([state_features, action_features], dim=-1)
        
        # Get Q-value
        q_value = self.combined_net(combined)
        
        return q_value

class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.001, buffer_size=100000, batch_size=64, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        
        # Initialize target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Initialize reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_momentum = 0.99
        
        # L2 regularization
        self.l2_reg = 1e-5
        
        # Parameter noise for exploration
        self.param_noise_std = 0.01
        self.param_noise_scale = 1.0
        self.param_noise_adaptation_rate = 0.01
        self.param_noise_distance = 0.0
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_reward_threshold = -300
        
    def select_action(self, state, training=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor, training=False)
        
        if training and random.random() < self.epsilon:
            noise = np.random.normal(0, 0.1, size=self.action_dim)
            action = action.squeeze().numpy() + noise
            action = np.clip(action, -1, 1)
        else:
            action = action.squeeze().numpy()
        
        # Apply parameter noise for exploration
        if training and random.random() < 0.1:  # 10% chance to use parameter noise
            action = self._apply_parameter_noise(state)
        
        return action
    
    def _apply_parameter_noise(self, state):
        # Create a copy of the actor with noisy parameters
        noisy_actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        noisy_actor.load_state_dict(self.actor.state_dict())
        
        # Add noise to parameters
        for param in noisy_actor.parameters():
            noise = torch.randn_like(param) * self.param_noise_std * self.param_noise_scale
            param.data.add_(noise)
        
        # Get action from noisy actor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = noisy_actor(state_tensor, training=False)
        
        return action.squeeze().numpy()
    
    def update(self, state, action, reward, next_state, done):
        # Update reward statistics
        self._update_reward_stats(reward)
        
        # Normalize reward
        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        # Store transition
        self.replay_buffer.append((state, action, normalized_reward, next_state, done))
        
        # Initialize losses
        actor_loss = 0.0
        critic_loss = 0.0
        
        # Only update if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return actor_loss, critic_loss
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch]))
        action_batch = torch.FloatTensor(np.array([x[1] for x in batch]))
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array([x[3] for x in batch]))
        done_batch = torch.FloatTensor(np.array([x[4] for x in batch])).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch, training=False)
            target_q = self.target_critic(next_state_batch, next_actions, training=False)
            target_q = reward_batch + (1 - done_batch) * self.gamma * target_q
        
        current_q = self.critic(state_batch, action_batch, training=True)
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Add L2 regularization
        for param in self.critic.parameters():
            critic_loss += self.l2_reg * torch.sum(param ** 2)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch, training=True), training=True).mean()
        
        # Add L2 regularization
        for param in self.actor.parameters():
            actor_loss += self.l2_reg * torch.sum(param ** 2)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target_networks()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update parameter noise if used
        if random.random() < 0.1:  # 10% chance to update parameter noise
            self._update_parameter_noise()
        
        return actor_loss.item(), critic_loss.item()
    
    def _update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def _update_reward_stats(self, reward):
        self.reward_mean = self.reward_momentum * self.reward_mean + (1 - self.reward_momentum) * reward
        self.reward_std = self.reward_momentum * self.reward_std + (1 - self.reward_momentum) * (reward - self.reward_mean) ** 2
        self.reward_std = np.sqrt(self.reward_std + 1e-8)
    
    def _update_parameter_noise(self):
        # Adapt parameter noise based on distance between actions
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = torch.FloatTensor([x[0] for x in batch])
        
        # Get actions from actor and noisy actor
        with torch.no_grad():
            actions = self.actor(state_batch, training=False)
            noisy_actions = self._apply_parameter_noise_batch(state_batch)
        
        # Calculate distance between actions
        distance = torch.mean(torch.abs(actions - noisy_actions)).item()
        
        # Update parameter noise scale
        if distance < self.param_noise_distance:
            self.param_noise_scale *= (1 + self.param_noise_adaptation_rate)
        else:
            self.param_noise_scale *= (1 - self.param_noise_adaptation_rate)
        
        # Clamp parameter noise scale
        self.param_noise_scale = np.clip(self.param_noise_scale, 0.01, 10.0)
    
    def _apply_parameter_noise_batch(self, state_batch):
        # Create a copy of the actor with noisy parameters
        noisy_actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        noisy_actor.load_state_dict(self.actor.state_dict())
        
        # Add noise to parameters
        for param in noisy_actor.parameters():
            noise = torch.randn_like(param) * self.param_noise_std * self.param_noise_scale
            param.data.add_(noise)
        
        # Get actions from noisy actor
        with torch.no_grad():
            actions = noisy_actor(state_batch, training=False)
        
        return actions
    
    def update_curriculum_stage(self, avg_reward):
        # Update curriculum stage based on average reward
        if avg_reward > self.curriculum_reward_threshold and self.curriculum_stage < 5:
            self.curriculum_stage += 1
            print(f"\nAdvancing to curriculum stage {self.curriculum_stage}")
            return True
        return False 