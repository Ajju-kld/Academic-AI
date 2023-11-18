import numpy as np
import time

class StudyScheduleEnvironment:
    def __init__(self, daily_time_quota):
        self.daily_time_quota = daily_time_quota
        self.state = daily_time_quota
        self.task_pool = [
    {'subject': 'Math', 'description': 'Algebra', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 5},
    {'subject': 'Physics', 'description': 'Mechanics', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Algebra', 'deadline': 4},
    {'subject': 'Physics', 'description': 'Electromagnetism', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Mechanics', 'deadline': 6},
    {'subject': 'Chemistry', 'description': 'Organic Chemistry', 'min_study_time': 2, 'max_study_time': 4, 'priority': 2, 'prerequisite': 'Electromagnetism', 'deadline': 8},
    {'subject': 'Chemistry', 'description': 'Inorganic Chemistry', 'min_study_time': 1, 'max_study_time': 3, 'priority': 2, 'prerequisite': 'Organic Chemistry', 'deadline': 7},
    {'subject': 'Biology', 'description': 'Cell Biology', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Biology', 'description': 'Genetics', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Cell Biology', 'deadline': 5},
    {'subject': 'Computer Science', 'description': 'Programming Fundamentals', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 7},
    {'subject': 'Computer Science', 'description': 'Data Structures', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Programming Fundamentals', 'deadline': 6},
    {'subject': 'Computer Science', 'description': 'Algorithms', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Data Structures', 'deadline': 8},
    {'subject': 'History', 'description': 'World War I', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'History', 'description': 'Renaissance', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'World War I', 'deadline': 5},
    {'subject': 'Literature', 'description': 'Shakespearean Plays', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 7},
    {'subject': 'Literature', 'description': 'Modern Poetry', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Shakespearean Plays', 'deadline': 6},
    {'subject': 'Art', 'description': 'Impressionism', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 8},
    {'subject': 'Art', 'description': 'Surrealism', 'min_study_time': 1, 'max_study_time': 3, 'priority': 2, 'prerequisite': 'Impressionism', 'deadline': 7},
    {'subject': 'Economics', 'description': 'Microeconomics', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Economics', 'description': 'Macroeconomics', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Microeconomics', 'deadline': 5},
    {'subject': 'Political Science', 'description': 'International Relations', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 7},
    {'subject': 'Political Science', 'description': 'Comparative Politics', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'International Relations', 'deadline': 6},
    {'subject': 'Psychology', 'description': 'Cognitive Psychology', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 8},
    {'subject': 'Psychology', 'description': 'Abnormal Psychology', 'min_study_time': 1, 'max_study_time': 3, 'priority': 2, 'prerequisite': 'Cognitive Psychology', 'deadline': 7},
    {'subject': 'Sociology', 'description': 'Social Movements', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Sociology', 'description': 'Cultural Sociology', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Social Movements', 'deadline': 5},
    {'subject': 'Geography', 'description': 'Human Geography', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 7},
    {'subject': 'Geography', 'description': 'Physical Geography', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Human Geography', 'deadline': 6},
    {'subject': 'Environmental Science', 'description': 'Climate Change', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 8},
    {'subject': 'Environmental Science', 'description': 'Biodiversity', 'min_study_time': 1, 'max_study_time': 3, 'priority': 2, 'prerequisite': 'Climate Change', 'deadline': 7},
    {'subject': 'Music', 'description': 'Classical Music', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Music', 'description': 'Jazz', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Classical Music', 'deadline': 5},
    {'subject': 'Language', 'description': 'Spanish', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 7},
    {'subject': 'Language', 'description': 'Chinese', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Spanish', 'deadline': 6},
    {'subject': 'Philosophy', 'description': 'Ethics', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 8},
    {'subject': 'Philosophy', 'description': 'Existentialism', 'min_study_time': 1, 'max_study_time': 3, 'priority': 2, 'prerequisite': 'Ethics', 'deadline': 7},
    {'subject': 'Health', 'description': 'Nutrition', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 6},
    {'subject': 'Health', 'description': 'Mental Health', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Nutrition', 'deadline': 5},
    {'subject': 'Engineering', 'description': 'Introduction to Engineering', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': None, 'deadline': 7},
    {'subject': 'Engineering', 'description': 'Computer Engineering', 'min_study_time': 3, 'max_study_time': 5, 'priority': 4, 'prerequisite': 'Introduction to Engineering', 'deadline': 6},
    {'subject': 'Engineering', 'description': 'Electrical Engineering', 'min_study_time': 2, 'max_study_time': 4, 'priority': 3, 'prerequisite': 'Computer Engineering', 'deadline': 8},
    # Add more topics as needed
]

        self.completed_topics = set()

    def reset(self):
        self.state = self.daily_time_quota
        self.completed_topics = set()
        return self.state

    def step(self, action):
        task = self.task_pool[action]
        
        # Check if the task has a prerequisite
        if task['prerequisite'] and task['prerequisite'] not in self.completed_topics:
            return self.state, 0  # Can't study this topic if the prerequisite is not completed

        # Check if the same subject topic has already been studied on the same day
        if task['subject'] in self.completed_topics:
            # Check if the deadline is close, allowing the same subject topic on the same day
            if task['deadline'] <= 2:  # You can adjust the threshold for a close deadline
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
import torch
import torch.nn as nn
import torch.optim as optim
seed = 42  # Choose any seed value
np.random.seed(seed)
torch.manual_seed(seed)
 
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QLearningAgent:
    def __init__(self, env, learning_rate, gamma, epsilon, num_epochs):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_epochs = num_epochs

        # Create Q-network
        self.q_network = QNetwork(1, len(env.task_pool))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.env.task_pool))
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor([state], dtype=torch.float32))
                
                return torch.argmax(q_values).item()

    def train(self):
        for epoch in range(self.num_epochs):
            state = self.env.reset()
            total_reward = 0
            self.env.completed_topics = set()  # Reset completed topics for each epoch

            while state > 0:
                action = self.select_action(state)
                next_state, reward = self.env.step(action)

                with torch.no_grad():
                    q_values_next = self.q_network(torch.tensor([next_state], dtype=torch.float32))
                    max_q_value_next = torch.max(q_values_next).item()

                target_q_value = reward + self.gamma * max_q_value_next
                current_q_value = self.q_network(torch.tensor([state], dtype=torch.float32))[action]

                loss = self.criterion(current_q_value, torch.tensor([target_q_value], dtype=torch.float32))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_reward += reward
                state = next_state
                if(state == 1):
                    break

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Total Reward: {total_reward}')

# Set parameters
daily_time_quota = 15
learning_rate = 0.75
gamma = 0.95
epsilon = 0.9
num_epochs = 50000

# Create environment and agent
env = StudyScheduleEnvironment(daily_time_quota)
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

    while state > 0:
        action = agent.select_action(state)
        task = env.task_pool[action]
        task_time = np.random.randint(task['min_study_time'], task['max_study_time'] + 1)

        if task_time <= state:
            schedule.append({'Subject': task['subject'], 'Task': task['description'], 'StudyTime': task_time})
            state -= task_time

    return schedule

# Generate a study schedule using the trained agent
final_schedule = generate_study_schedule(agent, env)

# Print the study schedule

fp=open("study.txt", "w")

for entry in final_schedule:
    print(f"Subject: {entry['Subject']}, Task: {entry['Task']}, Study Time: {entry['StudyTime']} hours")
    fp.write(f"Subject: {entry['Subject']}, Task: {entry['Task']}, Study Time: {entry['StudyTime']} hours")
fp.close()
