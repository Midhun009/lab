import numpy as np

grid_size = 5
num_actions = 4  
num_episodes = 1000
alpha = 0.1  
gamma = 0.9  
epsilon = 0.1  

Q = np.zeros((grid_size, grid_size, num_actions))

rewards = np.full((grid_size, grid_size), -1)
rewards[4, 4] = 10  
obstacles = [(1, 1), (2, 2), (3, 3)]  

actions = [(0, 1), (0, -1), (1, 0), (-1, 0)] 
# Q-learning algorithm
for episode in range(num_episodes):
    state = (0, 0)
    while state != (4, 4):  
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(Q[state[0], state[1]])

        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        if next_state in obstacles or next_state[0] < 0 or next_state[0] >= grid_size or next_state[1] < 0 or next_state[1] >= grid_size:
            reward = -5  
            next_state = state
        else:
            reward = rewards[next_state[0], next_state[1]]

        Q[state[0], state[1], action] = (1 - alpha) * Q[state[0], state[1], action] + alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]))

        state = next_state

state = (0, 0)
path = [state]
while state != (4, 4):
    action = np.argmax(Q[state[0], state[1]])
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    path.append(next_state)
    state = next_state

print("Optimal path:", path)
