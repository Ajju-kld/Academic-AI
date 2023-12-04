import gym
import numpy as np

# Define the task pool data structure
# Each task has a description, deadline, priority, minimum study time, maximum study time
task_pool = [
    {'description': 'Task A', 'deadline': 5, 'priority': 3, 'min_study_time': 2, 'max_study_time': 4},
    {'description': 'Task B', 'deadline': 3, 'priority': 4, 'min_study_time': 1, 'max_study_time': 3},
    # Add more tasks as needed
]

# Define the daily time quota
daily_time_quota = 8  # Maximum daily study time

# Define the RL environment
class TaskSchedulingEnv(gym.Env):
    def __init__(self):
        super(TaskSchedulingEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(len(task_pool))
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(1,))

        self.state = daily_time_quota  # Initial state (remaining time in a day)

    def reset(self):
        self.state = daily_time_quota
        return np.array([self.state])

    def step(self, action):
        task = task_pool[action]
        task_time = min(task['max_study_time'], self.state)
        reward = 0

        if task_time >= task['min_study_time']:
            reward = task['priority']

        self.state -= task_time

        # Check if the episode is done
        done = self.state == 0

        return np.array([self.state]), reward, done, {}

# Q-learning algorithm
from stable_baselines3 import QLearning

# Create the RL environment
env = TaskSchedulingEnv()

# Define Q-learning parameters
model = QLearning("MlpPolicy", env, verbose=1)

# Train the Q-learning agent
model.learn(total_timesteps=10000)
            
# Get the optimal schedule
schedule = []
state = daily_time_quota
while state > 0:
    action, _ = model.predict(np.array([state]))
    task = task_pool[action]
    task_time = min(task['max_study_time'], state)
    if task_time >= task['min_study_time']:
        schedule.append({'Task': task, 'StudyTime': task_time})
    state -= task_time

# Print the optimal schedule
for entry in schedule:
    print(f"Task: {entry['Task']['description']}, Study Time: {entry['StudyTime']}")
