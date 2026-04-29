import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output

class ParametricMaze:
    def __init__(self, width=10, height=10, wall_density=0.2, seed=None):
        if seed is not None: np.random.seed(seed)
        self.width = width
        self.height = height
        self.start_pos = (0, 0)
        self.goal_pos = (height - 1, width - 1)
        self.grid = np.zeros((height, width))

        # Posizionamento casuale dei muri
        for r in range(height):
            for c in range(width):
                if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                    if np.random.rand() < wall_density:
                        self.grid[r, c] = 1
        self.reset()

    def reset(self):
        self.current_pos = self.start_pos
        return self._to_state(self.current_pos)

    def _to_state(self, pos):
        return pos[0] * self.width + pos[1]

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        r, c = self.current_pos
        if action == 0: r_next, c_next = max(0, r - 1), c
        elif action == 1: r_next, c_next = min(self.height - 1, r + 1), c
        elif action == 2: r_next, c_next = r, max(0, c - 1)
        elif action == 3: r_next, c_next = r, min(self.width - 1, c + 1)

        if self.grid[r_next, c_next] == 1:
            reward, done = -5, False # Urto contro muro
        else:
            self.current_pos = (r_next, c_next)
            if self.current_pos == self.goal_pos:
                reward, done = 100, True
            else:
                reward, done = -1, False # Penalità per ogni passo

        return self._to_state(self.current_pos), reward, done

    def plot(self, path=None, title="Maze Visualization"):
        fig, ax = plt.subplots(figsize=(self.width/1.5, self.height/1.5))

        # Centratura e limiti
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

        for r in range(self.height):
            for c in range(self.width):
                y_plot = self.height - 1 - r
                if self.grid[r, c] == 1:
                    ax.add_patch(patches.Rectangle((c, y_plot), 1, 1, color='black'))
                elif (r, c) == self.start_pos:
                    ax.add_patch(patches.Rectangle((c, y_plot), 1, 1, color='blue', alpha=0.3))
                    ax.text(c+0.5, y_plot+0.5, 'START', ha='center', va='center', fontsize=8, weight='bold')
                elif (r, c) == self.goal_pos:
                    ax.add_patch(patches.Rectangle((c, y_plot), 1, 1, color='green', alpha=0.3))
                    ax.text(c+0.5, y_plot+0.5, 'GOAL', ha='center', va='center', fontsize=8, weight='bold')

        if path:
            y_coords = [self.height - 0.5 - p[0] for p in path]
            x_coords = [p[1] + 0.5 for p in path]
            ax.plot(x_coords, y_coords, color='red', linewidth=3, marker='o', markersize=4, zorder=5)

        plt.title(title)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

# PARAMETRI DEL LABIRINTO
WIDTH = 15
HEIGHT = 15
DENSITY = 0.25

# Generazione
env = ParametricMaze(WIDTH, HEIGHT, DENSITY, seed=314)

print("Labirinto generato.")
env.plot(title="Labirinto Iniziale")

class QAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0, decay=0.999):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha     # Learning Rate
        self.gamma = gamma     # Discount Factor
        self.epsilon = epsilon # Esplorazione iniziale al 100%
        self.decay = decay     # Riduzione graduale dell'esplorazione

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3) # Esplora
        return np.argmax(self.q_table[state]) # Sfrutta la conoscenza

    # using bellman equation to update q_table
    def update(self, s, a, r, s_next):
        target = r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s][a] += self.alpha * (target - self.q_table[s][a])

    def update_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.decay)
    
    # PARAMETRI TRAINING
EPISODES = 100
MAX_STEPS = 150000 # Limite per episodio per evitare loop infiniti
agent = QAgent(WIDTH * HEIGHT, 4)
reward_history = []
success_count = 0

print("Training in corso...")

for i in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        steps += 1
        if done: success_count += 1

    agent.update_epsilon()
    reward_history.append(total_reward)

    if (i+1) % 500 == 0:
        print(f"Ep {i+1}/{EPISODES} | Epsilon: {agent.epsilon:.2f} | Successi: {success_count}")

# 1. Grafico Apprendimento
plt.figure(figsize=(10, 4))
plt.plot(reward_history)
plt.title("Curva di Apprendimento")
plt.ylabel("Reward Totale")
plt.show()

# 2. Test Soluzione Finale
state = env.reset()
path = [env.start_pos]
done = False
steps = 0

while not done and steps < MAX_STEPS:
    action = np.argmax(agent.q_table[state])
    state, _, done = env.step(action)
    path.append(env.current_pos)
    steps += 1

if done:
    env.plot(path=path, title=f"Percorso Ottimale trovato in {steps} passi!")
else:
    print("ATTENZIONE: L'agente non ha trovato il Goal nel tempo limite. Prova ad aumentare EPISODES o ridurre DENSITY.")