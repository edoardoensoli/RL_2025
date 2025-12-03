import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class Policy(nn.Module):
    continuous = False  # Azioni discrete - più facile da apprendere

    def __init__(self, device=None):
        super(Policy, self).__init__()
        self.device = device if device else get_device()
        
        # 5 azioni discrete: niente, sinistra, destra, gas, freno
        self.n_actions = 5
        self.actions = [
            [0, 0, 0],      # 0: niente
            [-1, 0, 0],     # 1: sinistra
            [1, 0, 0],      # 2: destra
            [0, 1, 0],      # 3: gas
            [0, 0, 0.8],    # 4: freno
        ]
        
        # Hyperparameters
        self.lr = 2.5e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.c1 = 0.5
        self.c2 = 0.01
        self.batch_size = 64
        self.n_epochs = 4
        self.buffer_size = 1000
        self.max_grad_norm = 0.5
        
        # CNN
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc = nn.Linear(64 * 8 * 8, 512)
        
        # Actor (discrete) e Critic
        self.actor = nn.Linear(512, self.n_actions)
        self.critic = nn.Linear(512, 1)
        
        self._init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action_and_value(self, state, action=None):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if action is None:
            # Campiona azione
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        # Log prob dell'azione scelta - gestisce sia batch che singoli
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gestisci dimensioni
        if action.dim() == 1:
            log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        else:
            # action è 2D (batch, 1)
            action = action.squeeze(-1)
            log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        # Entropy
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action, log_prob, entropy, value
    
    def act(self, state):
        state = np.transpose(state, (2, 0, 1))
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.forward(state_t)
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
        return action_idx  # Ritorna l'indice per l'ambiente discreto

    def train(self):
        env = gym.make('CarRacing-v2', continuous=self.continuous)
        
        num_updates = 500
        best_reward = -float('inf')
        episode_rewards = []
        
        print(f"Training on device: {self.device}")
        print(f"Using {self.n_actions} discrete actions")
        
        for update in range(num_updates):
            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            values = []
            
            state, _ = env.reset()
            episode_reward = 0
            episodes_in_buffer = 0
            
            for step in range(self.buffer_size):
                state_t = torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_idx, log_prob, _, value = self.get_action_and_value(state_t)
                
                env_action = action_idx.item()  # L'ambiente discreto vuole l'indice
                next_state, reward, terminated, truncated, _ = env.step(env_action)
                done = terminated or truncated
                
                states.append(state_t)
                actions.append(action_idx)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    episode_rewards.append(episode_reward)
                    episodes_in_buffer += 1
                    episode_reward = 0
                    state, _ = env.reset()
            
            # GAE
            with torch.no_grad():
                state_t = torch.tensor(np.transpose(state, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.device)
                _, _, _, next_value = self.get_action_and_value(state_t)
                
                values_t = torch.cat(values).squeeze()
                next_value = next_value.squeeze()
                
                advantages = torch.zeros(self.buffer_size, device=self.device)
                last_gae_lam = 0
                
                for t in reversed(range(self.buffer_size)):
                    if t == self.buffer_size - 1:
                        nextnonterminal = 1.0 - float(done)
                        nextval = next_value
                    else:
                        nextnonterminal = 1.0 - float(dones[t])
                        nextval = values_t[t + 1]
                    
                    delta = rewards[t] + self.gamma * nextval * nextnonterminal - values_t[t]
                    last_gae_lam = delta + self.gamma * self.gae_lambda * nextnonterminal * last_gae_lam
                    advantages[t] = last_gae_lam
                
                returns = advantages + values_t

            b_states = torch.cat(states)
            b_actions = torch.cat(actions)  # Concatena invece di stack
            b_log_probs = torch.cat(log_probs)
            b_returns = returns
            b_advantages = advantages
            
            # PPO Update
            b_inds = np.arange(self.buffer_size)
            for epoch in range(self.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.buffer_size, self.batch_size):
                    end = start + self.batch_size
                    mb_inds = b_inds[start:end]
                    
                    _, new_log_prob, entropy, new_value = self.get_action_and_value(
                        b_states[mb_inds], b_actions[mb_inds]
                    )
                    
                    logratio = new_log_prob - b_log_probs[mb_inds]
                    ratio = logratio.exp()
                    
                    mb_adv = b_advantages[mb_inds]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    v_loss = 0.5 * ((new_value.squeeze() - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    
                    loss = pg_loss + self.c1 * v_loss - self.c2 * entropy_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()
            
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else (np.mean(episode_rewards) if episode_rewards else 0)
            print(f"Update {update+1}/{num_updates} | Ep: {episodes_in_buffer} | Avg(10): {avg_reward:.1f} | Best: {best_reward:.1f}")
            
            if len(episode_rewards) >= 10 and avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
                print(f"  -> Best model! Reward: {best_reward:.1f}")
            
            if (update + 1) % 50 == 0:
                self.save()
                
            if best_reward > 700:
                print("Good enough!")
                break
        
        env.close()
        print(f"Done. Best: {best_reward:.1f}")

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device, weights_only=True))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
