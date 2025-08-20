import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Add Technical Indicators
def add_technical_indicators(data):
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI using EMA
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)
    return data.reset_index(drop=True)

# Step 2: Environment
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
        self.total_value = self.balance
        self.actions = []
        return self._get_state()

    def _get_state(self):
        slice_ = slice(self.current_step - self.window_size, self.current_step)
        state = np.concatenate([
            self.data['Open'][slice_].values,
            self.data['High'][slice_].values,
            self.data['Low'][slice_].values,
            self.data['Close'][slice_].values,
            self.data['SMA_10'][slice_].values,
            self.data['EMA_10'][slice_].values,
            self.data['MACD'][slice_].values,
            self.data['Signal_Line'][slice_].values,
            self.data['RSI'][slice_].values
        ])
        return np.round(state, 2)

    def step(self, action):
        done = False
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True

        current_price = self.data['Close'][self.current_step]
        previous_total_value = self.balance + self.stock_held * current_price

        # Execute action
        if action == 0 and self.balance >= current_price:
            self.stock_held += 1
            self.balance -= current_price
            self.actions.append(('Buy', self.current_step, current_price))
        elif action == 1 and self.stock_held > 0:
            self.stock_held -= 1
            self.balance += current_price
            self.actions.append(('Sell', self.current_step, current_price))
        else:
            self.actions.append(('Hold', self.current_step, current_price))

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        new_price = self.data['Close'][self.current_step]
        self.total_value = self.balance + self.stock_held * new_price
        reward = self.total_value - previous_total_value

        return self._get_state(), reward, done

# Step 3: Q-Learning Agent
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

# Step 4: Training Function
def train(agent, env, n_episodes=1000):
    episode_rewards = []
    total_values = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_total_value = []
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            episode_total_value.append(env.total_value)
        episode_rewards.append(total_reward)
        total_values.append(episode_total_value)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
    return total_values

# Step 5: Buy-and-Hold Strategy
def buy_and_hold(data):
    initial_balance = 10000
    initial_price = data['Close'].iloc[0]
    stock_held = initial_balance // initial_price
    remaining_cash = initial_balance % initial_price
    return remaining_cash + stock_held * data['Close'].values

# Step 6: Plotting
def plot_trading_results(env, total_values, data, stock_symbol="FCPO"):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'].values, label='Stock Price', color='blue')

    # Actions
    buy_points = [(step, price) for action, step, price in env.actions if action == 'Buy']
    sell_points = [(step, price) for action, step, price in env.actions if action == 'Sell']
    if buy_points:
        steps, prices = zip(*buy_points)
        plt.scatter(steps, prices, marker='^', color='green', label='Buy', alpha=1)
    if sell_points:
        steps, prices = zip(*sell_points)
        plt.scatter(steps, prices, marker='v', color='red', label='Sell', alpha=1)

    # Q-Learning Agent performance
    agent_total_value = np.mean([np.array(v) for v in total_values], axis=0)
    plt.plot(range(len(agent_total_value)), agent_total_value, label='Q-Learning Agent Performance', color='orange')

    # Buy & hold
    buy_hold_value = buy_and_hold(data)
    plt.plot(buy_hold_value, label='Buy and Hold', color='green', linestyle='--')

    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.title(f'Trading Actions for {stock_symbol} (Q-Learning vs Buy & Hold)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    try:
        data = pd.read_excel("KPONF1 hourly 1.xlsx")
        if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            print("Error: Data must have 'Open', 'High', 'Low', and 'Close' columns.")
        else:
            env = StockTradingEnv(data, window_size=10)
            agent = QLearningAgent(n_actions=3)
            total_values = train(agent, env, n_episodes=500)
            plot_trading_results(env, total_values, env.data)
    except Exception as e:
        print("Failed to load or process data:", str(e))
