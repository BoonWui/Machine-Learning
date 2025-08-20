import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define the Environment
class StockTradingEnv:
    def __init__(self, data, window_size=10):
        self.data = data
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = 10000  # Starting with $10,000
        self.balance = self.initial_balance
        self.stock_held = 0
        self.total_value = self.balance + self.stock_held * self.data['Close'][self.current_step]

    def reset(self):
        self.balance = self.initial_balance
        self.stock_held = 0
        self.current_step = self.window_size
        self.total_value = self.balance + self.stock_held * self.data['Close'][self.current_step]
        return self._get_state()

    def _get_state(self):
        # Get a window of historical stock prices
        return self.data['Close'][self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        current_price = self.data['Close'][self.current_step]
        reward = 0
        done = False

        if action == 0:  # Buy
            if self.balance >= current_price:
                self.stock_held += 1
                self.balance -= current_price
        elif action == 1:  # Sell
            if self.stock_held > 0:
                self.stock_held -= 1
                self.balance += current_price
        elif action == 2:  # Hold
            pass

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        self.total_value = self.balance + self.stock_held * self.data['Close'][self.current_step]
        reward = self.total_value - (self.balance + self.stock_held * self.data['Close'][self.current_step - 1])

        return self._get_state(), reward, done

# Step 2: Define the Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax([self.get_q_value(next_state, a) for a in range(self.n_actions)])
        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
        td_error = td_target - self.get_q_value(state, action)
        new_q_value = self.get_q_value(state, action) + self.learning_rate * td_error
        self.q_table[(tuple(state), action)] = new_q_value

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)

# Step 3: Train the Agent
def train(agent, env, n_episodes=1000):
    episode_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    return episode_rewards

# Step 4: Visualization and Evaluation
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance Over Time')
    plt.show()

# Main
if __name__ == "__main__":
    # Load data from Excel
    data = pd.read_excel("KPONF1 hourly 1.xlsx")
    # Assuming the Excel file has a 'Close' column. You may need to adjust if the column is named differently.
    # Example: print(data.head()) to check the data structure.
    
    # Ensure the data contains the 'Close' column
    if 'Close' not in data.columns:
        raise ValueError("The data must contain a 'Close' column.")

    env = StockTradingEnv(data=data, window_size=10)  # Use the data loaded from Excel
    agent = QLearningAgent(n_actions=3)  # Actions: 0 = Buy, 1 = Sell, 2 = Hold
    rewards = train(agent, env, n_episodes=1000)
    plot_rewards(rewards)
