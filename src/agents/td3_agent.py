import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device="cpu", lr=3e-4):
        self.device = device
        self.max_action = max_action
        self.total_it = 0

        # Actor
        self.actor = self._build_actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = self._build_critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

    def _build_actor(self, s_dim, a_dim, max_action):
        return nn.Sequential(
            nn.Linear(s_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, a_dim), nn.Tanh()
        )

    def _build_critic(self, s_dim, a_dim):
        # We need a custom module because TD3 has two heads (Q1, Q2)
        class TwinCritic(nn.Module):
            def __init__(self):
                super().__init__()
                # Q1
                self.l1 = nn.Linear(s_dim + a_dim, 256)
                self.l2 = nn.Linear(256, 256)
                self.l3 = nn.Linear(256, 1)
                # Q2
                self.l4 = nn.Linear(s_dim + a_dim, 256)
                self.l5 = nn.Linear(256, 256)
                self.l6 = nn.Linear(256, 1)

            def forward(self, state, action):
                sa = torch.cat([state, action], 1)
                q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
                q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
                return q1, q2
            
            def Q1(self, state, action):
                sa = torch.cat([state, action], 1)
                return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))

        return TwinCritic()

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise * self.max_action, size=action.shape))
        return action.clip(-self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005, noise_clip=0.5, policy_noise=0.2):
        self.total_it += 1
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None
        if self.total_it % 2 == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        return critic_loss.item(), (actor_loss.item() if actor_loss else None)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location="cpu"))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location="cpu"))