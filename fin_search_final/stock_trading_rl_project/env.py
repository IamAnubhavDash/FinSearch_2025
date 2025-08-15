import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(TradingEnv, self).__init__()

        self.data = data
        self.max_steps = len(data) - 1
        self.current_step = 0

        # Actions: 0 = Buy, 1 = Hold, 2 = Sell
        self.action_space = spaces.Discrete(3)
        # Observations: feature vector size (e.g., 8 features)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.data.shape[1],), dtype=np.float32)

        self.position = 0  # 0 = no position, 1 = holding long
        self.entry_price = 0
        self.total_profit = 0

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        return self.data[self.current_step]

    def step(self, action):
        done = False
        reward = 0

        current_price = self.data[self.current_step][3]  # Close price (index 3 in features)

        # Execute action
        if action == 0:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
                reward = 0
        elif action == 2:  # Sell
            if self.position == 1:
                reward = current_price - self.entry_price
                self.total_profit += reward
                self.position = 0
                self.entry_price = 0
        # else Hold: reward = 0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self.data[self.current_step] if not done else np.zeros(self.data.shape[1])

        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Position: {self.position}, Total Profit: {self.total_profit:.4f}')
