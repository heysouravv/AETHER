import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, *args):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(*args))
        else:
            self.buffer[self.position] = Experience(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[:len(self.buffer)]
        probs = probs ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class ImprovedDuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImprovedDuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU()
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

class DroneDodgeSimulation:
    def __init__(self):
        self.field_size = np.array([10000.0, 10000.0, 5000.0])  # x, y, z
        self.start_pos = self.field_size / 2
        self.drone_pos = self.start_pos.copy()
        self.projectiles = []
        self.drone_velocity = np.zeros(3)
        self.max_drone_speed = 300.0
        self.min_drone_speed = 50.0
        self.max_projectile_speed = 3000.0
        self.min_projectile_speed = 2000.0
        self.max_projectiles = 5
        self.projectile_spawn_rate = 0.05
        self.time_step = 1/30
        self.hit_count = 0
        self.drone_radius = 50.0
        self.min_altitude = 500.0  # Minimum safe altitude
        
        # Actions: (x, y, z) directions, each can be -1, 0, or 1
        self.actions = [(x, y, z) for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1]]
        self.actions.remove((0, 0, 0))  # Remove the "do nothing" action
        
        self.state_dim = 39  # 3 (drone pos) + 3 (drone vel) + 3 (relative center) + 5 * 6 (projectile info)
        self.action_dim = len(self.actions)
        self.dqn = ImprovedDuelingDQN(self.state_dim, self.action_dim)
        self.target_dqn = ImprovedDuelingDQN(self.state_dim, self.action_dim)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.learning_rate = 0.0001
        self.batch_size = 128
        self.update_target_every = 1000
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        
        self.memory = PrioritizedReplayBuffer(50000)
        self.steps = 0
        self.episode_reward = 0
        self.n_step = 3
        self.n_step_buffer = deque(maxlen=self.n_step)

    def get_state(self):
        projectile_info = []
        for p in sorted(self.projectiles, key=lambda x: np.linalg.norm(x['pos'] - self.drone_pos))[:5]:
            relative_pos = p['pos'] - self.drone_pos
            relative_vel = p['velocity'] - self.drone_velocity
            projectile_info.extend(relative_pos)
            projectile_info.extend(relative_vel)
        projectile_info.extend([0] * 6 * (5 - len(projectile_info) // 6))
        
        relative_center = self.start_pos - self.drone_pos
        return np.concatenate([
            self.drone_pos,
            self.drone_velocity,
            relative_center,
            projectile_info
        ])

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.dqn(state_tensor)
        return q_values.argmax().item()

    def spawn_projectile(self):
        if len(self.projectiles) < self.max_projectiles and np.random.random() < self.projectile_spawn_rate:
            side = np.random.choice(['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'])
            pos = np.random.rand(3) * self.field_size
            if side == 'x_min': pos[0] = 0
            elif side == 'x_max': pos[0] = self.field_size[0]
            elif side == 'y_min': pos[1] = 0
            elif side == 'y_max': pos[1] = self.field_size[1]
            elif side == 'z_min': pos[2] = 0
            else: pos[2] = self.field_size[2]
            
            direction = self.drone_pos - pos
            direction /= np.linalg.norm(direction)
            speed = np.random.uniform(self.min_projectile_speed, self.max_projectile_speed)
            velocity = direction * speed
            
            self.projectiles.append({
                'pos': pos,
                'velocity': velocity
            })

    def calculate_reward(self, old_pos, new_pos, projectiles, hit):
        reward = 0.1  # Base reward for surviving

        if hit:
            reward -= 50  # Penalty for getting hit

        # Reward for dodging projectiles
        for projectile in projectiles:
            distance = np.linalg.norm(projectile['pos'] - new_pos)
            if distance < self.drone_radius * 3:
                reward += 5 * (1 - distance / (self.drone_radius * 3))  # Scaled reward for near misses

        # Penalty for being too far from the center
        distance_from_center = np.linalg.norm(new_pos[:2] - self.start_pos[:2])  # Only consider x and y
        position_penalty = -distance_from_center / 5000  # Reduced penalty
        reward += position_penalty

        # Penalty for being too close to the ground
        if new_pos[2] < self.min_altitude:
            altitude_penalty = -10 * (1 - new_pos[2] / self.min_altitude)
            reward += altitude_penalty

        # Small penalty for excessive movement
        movement = np.linalg.norm(new_pos - old_pos)
        movement_penalty = -movement / 50
        reward += movement_penalty

        return reward

    def calculate_n_step_returns(self):
        reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += r * (self.gamma ** i)
        return reward

    def update(self):
        state = self.get_state()
        action = self.get_action(state)
        
        move = np.array(self.actions[action])
        speed = np.random.uniform(self.min_drone_speed, self.max_drone_speed)
        self.drone_velocity = move * speed
        old_pos = self.drone_pos.copy()
        new_pos = self.drone_pos + self.drone_velocity * self.time_step
        
        # Ensure the drone stays above the minimum altitude
        new_pos[2] = max(new_pos[2], self.min_altitude)
        
        self.drone_pos = np.clip(new_pos, [0, 0, self.min_altitude], self.field_size)

        self.spawn_projectile()

        hit = False
        for projectile in self.projectiles:
            projectile['pos'] += projectile['velocity'] * self.time_step
            if np.linalg.norm(projectile['pos'] - self.drone_pos) <= self.drone_radius:
                self.hit_count += 1
                hit = True
                projectile['pos'] = np.array([-1000, -1000, -1000])

        reward = self.calculate_reward(old_pos, self.drone_pos, self.projectiles, hit)

        self.projectiles = [p for p in self.projectiles if np.all((0 <= p['pos']) & (p['pos'] <= self.field_size))]

        new_state = self.get_state()
        
        self.n_step_buffer.append((state, action, reward, new_state, False))
        
        if len(self.n_step_buffer) == self.n_step:
            n_step_state, n_step_action, _, _, _ = self.n_step_buffer[0]
            n_step_reward = self.calculate_n_step_returns()
            self.memory.push(n_step_state, n_step_action, n_step_reward, new_state, False)
        
        if len(self.memory) > self.batch_size:
            self.train_model()

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.episode_reward += reward

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.drone_pos, [p['pos'] for p in self.projectiles], self.hit_count, self.episode_reward


    def train_model(self):
        experiences, indices, weights = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Optimize: Use numpy.vstack to efficiently combine states and next_states
        states = np.vstack(batch.state)
        next_states = np.vstack(batch.next_state)

        # Convert to tensors after stacking
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(batch.done)

        # Double DQN
        next_actions = self.dqn(next_states).max(1)[1].unsqueeze(1)
        next_q_values = self.target_dqn(next_states).gather(1, next_actions).squeeze(1)
        
        target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)
        current_q_values = self.dqn(states).gather(1, actions).squeeze(1)

        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach(), reduction='none')
        loss = (loss * torch.FloatTensor(weights)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        # Update priorities
        priorities = (current_q_values - target_q_values).abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, priorities)


# Set up the simulation
sim = DroneDodgeSimulation()

# Set up the 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, sim.field_size[0])
ax.set_ylim(0, sim.field_size[1])
ax.set_zlim(0, sim.field_size[2])
ax.set_title("Altitude-Aware 6-Axis Drone DQN-based Dodge Simulation")
ax.set_xlabel("X Distance (feet)")
ax.set_ylabel("Y Distance (feet)")
ax.set_zlabel("Z Distance (feet)")
drone_point, = ax.plot([], [], [], 'bo', markersize=10, label='Drone')
projectile_points = [ax.plot([], [], [], 'ro', markersize=5)[0] for _ in range(sim.max_projectiles)]
hit_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
reward_text = ax.text2D(0.02, 0.94, '', transform=ax.transAxes)
ax.legend()

# Add a plane to represent the minimum altitude
x = np.linspace(0, sim.field_size[0], 2)
y = np.linspace(0, sim.field_size[1], 2)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, sim.min_altitude)
ax.plot_surface(X, Y, Z, alpha=0.2, color='g')

def animate(frame):
    drone_pos, projectile_positions, hit_count, episode_reward = sim.update()
    
    drone_point.set_data([drone_pos[0]], [drone_pos[1]])
    drone_point.set_3d_properties([drone_pos[2]])
    
    for i, proj_pos in enumerate(projectile_positions):
        if i < len(projectile_points):
            projectile_points[i].set_data([proj_pos[0]], [proj_pos[1]])
            projectile_points[i].set_3d_properties([proj_pos[2]])
            projectile_points[i].set_visible(True)
        else:
            break
    
    for i in range(len(projectile_positions), len(projectile_points)):
        projectile_points[i].set_visible(False)
    
    hit_text.set_text(f'Hits: {hit_count}')
    reward_text.set_text(f'Episode Reward: {episode_reward:.0f}')
    
    return [drone_point, hit_text, reward_text] + projectile_points

# Create the animation
anim = FuncAnimation(fig, animate, frames=5000, interval=33, blit=False)

plt.show()