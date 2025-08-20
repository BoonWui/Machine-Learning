import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
import random

# Step 1: Add Technical Indicators
def add_technical_indicators(data):
    data = data.copy()
    data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))

    data = data.dropna().reset_index(drop=True)
    return data

# Step 2: Define the Environment
class StockTradingEnv:
    def __init__(self, data, window_size=10):
        self.data = add_technical_indicators(data)
        self.window_size = window_size
        self.initial_balance = 10000
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.stock_held = 0
        self.current_step = self.window_size
        self.actions = []
        return self._get_state()

    def _get_state(self):
        start = self.current_step - self.window_size
        end = self.current_step
        state = np.concatenate([
            self.data['Open'][start:end].values,
            self.data['High'][start:end].values,
            self.data['Low'][start:end].values,
            self.data['Close'][start:end].values,
            self.data['SMA_10'][start:end].values,
            self.data['EMA_10'][start:end].values,
            self.data['MACD'][start:end].values,
            self.data['Signal_Line'][start:end].values,
            self.data['RSI'][start:end].values,
        ])
        return state

    def step(self, action):
        price_now = self.data['Close'][self.current_step]
        done = False

        # Actions: Buy (0), Sell (1), Hold (2)
        if action == 0 and self.balance >= price_now:
            self.stock_held += 1
            self.balance -= price_now
            self.actions.append(('Buy', self.current_step, price_now))
        elif action == 1 and self.stock_held > 0:
            self.stock_held -= 1
            self.balance += price_now
            self.actions.append(('Sell', self.current_step, price_now))
        else:
            self.actions.append(('Hold', self.current_step, price_now))

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        price_next = self.data['Close'][self.current_step]
        total_value_next = self.balance + self.stock_held * price_next
        total_value_now = self.balance + self.stock_held * price_now
        reward = total_value_next - total_value_now

        return self._get_state(), reward, done, total_value_next

# Step 3: Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        next_states = np.array(next_states)
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q_values[i])
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Step 4: Train the Agent
def train(agent, env, n_episodes=1000):
    episode_rewards = []
    total_values = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_total_value = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, total_value = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            episode_total_value.append(total_value)

        episode_rewards.append(total_reward)
        total_values.append(episode_total_value)

        # Print progress every 100 episodes with actual episode info
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

        # Update target model every few episodes
        if episode % 10 == 0:
            agent.update_target_model()

    return total_values

# Step 5: Buy and Hold Strategy
def buy_and_hold(data):
    initial_balance = 10000
    initial_price = data['Close'].iloc[0]
    stock_held = initial_balance // initial_price
    remaining_balance = initial_balance % initial_price
    total_value = remaining_balance + stock_held * data['Close'].values
    return total_value

# Step 6: Plot Trading Results
def plot_trading_results(env, total_values, data, stock_symbol="FCPO"):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Stock Price', color='blue')

    buy_points = [(step, price) for action, step, price in env.actions if action == 'Buy']
    sell_points = [(step, price) for action, step, price in env.actions if action == 'Sell']

    if buy_points:
        buy_steps, buy_prices = zip(*buy_points)
        plt.scatter(buy_steps, buy_prices, marker='^', color='green', label='Buy', alpha=1)
    if sell_points:
        sell_steps, sell_prices = zip(*sell_points)
        plt.scatter(sell_steps, sell_prices, marker='v', color='red', label='Sell', alpha=1)

    agent_total_value = np.array(total_values).mean(axis=0)
    plt.plot(agent_total_value, label='DQN Agent Performance', color='orange')

    buy_hold_value = buy_and_hold(data)
    plt.plot(buy_hold_value, label='Buy and Hold', color='green', linestyle='--')

    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.title(f'Trading Performance for {stock_symbol} (DQN vs Buy & Hold)')
    plt.legend()
    plt.show()

# Main
if __name__ == "__main__":
    # Load your data
    data = pd.read_excel("KPONF1 hourly 1.xlsx")  # Replace with your correct file path
    required_columns = ['Open', 'High', 'Low', 'Close']
    
    if not all(col in data.columns for col in required_columns):
        print("Error: Data must include 'Open', 'High', 'Low', and 'Close' columns.")
    else:
        env = StockTradingEnv(data=data, window_size=10)
        state_size = len(env._get_state())
        action_size = 3  # Buy, Sell, Hold
        agent = DQNAgent(state_size, action_size)
        total_values = train(agent, env, n_episodes=500)
        plot_trading_results(env, total_values, env.data)
