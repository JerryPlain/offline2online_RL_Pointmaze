import torch
import torch.nn.functional as F
from .td3_agent import TD3Agent

class TD3BCAgent(TD3Agent):
    def __init__(self, state_dim, action_dim, max_action,
                 device="cpu", lr=3e-4,
                 bc_coef=2.5, policy_delay=2):
        super().__init__(state_dim, action_dim, max_action, device=device, lr=lr)
        self.bc_coef = bc_coef          # BC loss weights
        self.policy_delay = policy_delay

    def train(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # ---------- Critic update ----------
        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None

        # ---------- Actor + Target net（Delayed update） ----------
        if self.total_it % self.policy_delay == 0:
            #TD3：maxmize Q(s, π(s))
            pi = self.actor(state)
            q_pi, _ = self.critic(state, pi)
            td3_actor_loss = -q_pi.mean()

            # BC
            bc_loss = F.mse_loss(pi, action)

            # TD3-BC：combine
            actor_loss = td3_actor_loss + self.bc_coef * bc_loss

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # soft update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return critic_loss.item(), actor_loss.item() if actor_loss is not None else None
