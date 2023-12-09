import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers, models

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
    def is_episode_done(self):
        # Check if the available study time for the day is exhausted
        return self.state <= 0  

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

seed = 42  # Choose any seed value
np.random.seed(seed)
# torch.manual_seed(seed)
 
class QNetwork(models.Model):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(output_size)

    def call(self, x):
        # Add a check for 1D input and reshape if necessary
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)

        x = self.fc1(x)
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.criterion = tf.keras.losses.MeanSquaredError()

    def select_action(self, state):
        f = np.random.rand()
        if f < self.epsilon:
            return np.random.choice(len(self.env.task_pool))
        else:
            state = np.expand_dims(state, axis=0)  # Add a batch dimension
            q_values = self.q_network(state.astype(np.float32))
            return np.argmax(q_values)

    def train(self):
        for epoch in range(self.num_epochs):
            state = self.env.reset()
            total_reward = 0
            self.env.completed_topics = set()  # Reset completed topics for each epoch
            max_steps = 75  # Adjust as needed
            step_count = 0
            with tf.device('/GPU:0'):
                while step_count < max_steps:
                    action = self.select_action(state)
                    next_state, reward = self.env.step(action)

                    with tf.GradientTape() as tape:
                        q_values_next = self.q_network(np.array([next_state], dtype=np.float32))
                        max_q_value_next = tf.reduce_max(q_values_next)

                        target_q_value = reward + self.gamma * max_q_value_next
                        current_q_value = self.q_network(np.array([state], dtype=np.float32))[0, action]

                        loss = self.criterion(tf.expand_dims(current_q_value, 0), tf.expand_dims(target_q_value, 0))

                    gradients = tape.gradient(loss, self.q_network.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

                    total_reward += reward
                    state = next_state
                    step_count += 1
                    if self.env.is_episode_done():
                        break

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Total Reward: {total_reward}')

        # Save the model
        self.q_network.save_weights("qnetwork.h5")
    def load_model(self, filepath):
        # Load the model weights
        self.q_network.load_weights(filepath)

# Set parameters
daily_time_quota = 12  # Assuming 12 hours available for study each day
learning_rate = 0.01
gamma = 0.9
epsilon = 0.3
num_epochs = 150

# Create environment and agent
env = StudyScheduleEnvironment(daily_time_quota)
agent = QLearningAgent(env, learning_rate, gamma, epsilon, num_epochs)
start_time = time.time()
print(agent.q_network)
# Train the agent
agent.train()
agent.load_model("./qnetwork.h5")

training_time = time.time() - start_time
print(f"Training completed successfully. Training Time: {training_time} seconds")


def generate_study_schedule(agent, env):
    state = env.reset()
    schedule = []
    selected_tasks = set()

    while state > 0:
        action = agent.select_action(state)

        # Check if the task has already been selected
        if action in selected_tasks:
            print("Task has already been selected")
            continue

        task = env.task_pool[action]
        task_time = np.random.randint(task['min_study_time'], task['max_study_time'] + 1)

        if task_time <= state:
            schedule.append({'Subject': task['subject'], 'Task': task['description'], 'StudyTime': task_time})
            state -= task_time

            # Add the index to the set of selected tasks
            selected_tasks.add(action)

    return schedule


# Generate a study schedule using the trained agent
final_schedule = generate_study_schedule(agent, env)

# Print the study schedule

fp=open("study.txt", "w")

for entry in final_schedule:
    print(f"Subject: {entry['Subject']}, Task: {entry['Task']}, Study Time: {entry['StudyTime']} hours")
    fp.write(f"Subject: {entry['Subject']}, Task: {entry['Task']}, Study Time: {entry['StudyTime']} hours")
fp.close()
