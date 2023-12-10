import numpy as np
import pandas as pd
import time

class StudyScheduleEnvironment:
    def __init__(self, daily_time_quota, task_pool):
        self.daily_time_quota = daily_time_quota
        self.state = daily_time_quota
        self.task_pool = task_pool
        self.completed_topics = set()

    def reset(self):
        self.state = self.daily_time_quota
        self.completed_topics = set()
        return self.state

    def is_episode_done(self):
        return self.state <= 0

    def step(self, action):
        task = self.task_pool[action]
        
        if task['prerequisite'] and task['prerequisite'] not in self.completed_topics:
            return self.state, 0  # Can't study this topic if the prerequisite is not completed

        if task['subject'] in self.completed_topics:
            if task['deadline'] <= 2:
                pass
            else:
                return self.state, 0  # Can't study the same subject on the same day

        task_time = np.random.randint(task['min_study_time'], task['max_study_time'] + 1)
        reward = 0

        if task_time <= self.state:
            self.state -= task_time
            reward = task['priority']
            
            self.completed_topics.add(task['subject'])

        return self.state, reward


class QLearningAgent:
    def __init__(self, env, learning_rate, gamma, epsilon, num_epochs):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_epochs = num_epochs

        # Create Q-table
        self.q_table = np.zeros((len(env.task_pool),))
        self.actions = np.arange(len(env.task_pool))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table)

    def train(self):
        for epoch in range(self.num_epochs):
            state = self.env.reset()
            total_reward = 0

            self.env.completed_topics = set()  # Reset completed topics for each epoch
            max_steps = 1000  # Adjust as needed
            step_count = 0

            while step_count < max_steps:
                action = self.select_action(state)
                next_state, reward = self.env.step(action)

                max_q_value_next = np.max(self.q_table)

                target_q_value = reward + self.gamma * max_q_value_next
                current_q_value = self.q_table[action]

                self.q_table[action] += self.learning_rate * (target_q_value - current_q_value)

                total_reward += reward
                state = next_state
                step_count += 1
                

                if self.env.is_episode_done():
                    break

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Total Reward: {total_reward}')
            # print('Q-values:', self.q_table)

        print("Q-table:")
        print(self.q_table)


# Set parameters
daily_time_quota = 12
learning_rate = 0.01
gamma = 0.9
epsilon = 0.3
num_epochs = 150

# Read task pool from CSV file
task_pool_df = pd.read_csv('data.csv')
task_pool = task_pool_df.to_dict('records')
print('Task pool:', task_pool)

# Create environment and agent
env = StudyScheduleEnvironment(daily_time_quota, task_pool)
agent = QLearningAgent(env, learning_rate, gamma, epsilon, num_epochs)
start_time = time.time()
print("reached the training phase")
# Train the agent
agent.train()

training_time = time.time() - start_time
print(f"Training completed successfully. Training Time: {training_time} seconds")


def generate_study_schedule(agent, env):
    state = env.reset()
    schedule = []
    selected_tasks = set()

    while state > 0:
        action = agent.select_action(state)

        if action in selected_tasks:
            continue

        task = env.task_pool[action]
        task_time = np.random.randint(task['min_study_time'], task['max_study_time'] + 1)

        if task_time <= state:
            schedule.append({'Subject': task['subject'], 'Task': task['description'], 'StudyTime': task_time})
            state -= task_time
            selected_tasks.add(action)

    return schedule


final_schedule = generate_study_schedule(agent, env)

fp = open("study.txt", "w")

for entry in final_schedule:
    print(f"Subject: {entry['Subject']}, Task: {entry['Task']}, Study Time: {entry['StudyTime']} hours")
    fp.write(f"Subject: {entry['Subject']}, Task: {entry['Task']}, Study Time: {entry['StudyTime']} hours")

fp.close()
