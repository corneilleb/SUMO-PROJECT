import os
import sys
import json
import time
import os
import optparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sumolib import checkBinary
import traci
import matplotlib.pyplot as plt


def load_config(file_path):  # (Line 12)
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Config file '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Config file '{file_path}' is not a valid JSON file.")
        sys.exit(1)


def ensure_dir(directory):
    if directory:  # Check if directory is NOT empty
        if not os.path.exists(directory):
            os.makedirs(directory)
    else:
        print("Error: Directory path is empty.")


class SumoEnvironment:
    def __init__(self, config):
        self.config = config
        if config['gui']:
            self.sumo_binary = checkBinary("sumo-gui")
        else:
            self.sumo_binary = checkBinary("sumo")
        self.epochs = config['epochs']
        self.steps = config['steps']
        self.junctions = []

    def start_simulation(self):
        try:
            traci.start([self.sumo_binary, "-c", self.config['sumo_config'], "--tripinfo-output", self.config['tripinfo_output']])
            self.junctions = traci.trafficlight.getIDList()
            if not self.junctions:
                raise ValueError("No traffic light junctions detected in SUMO simulation.")
            print("Simulation started.")
        except traci.exceptions.FatalTraCIError as e:
            print(f"Error starting SUMO simulation: {e}")
            sys.exit(1)
        except ValueError as ve:
            print(f"Error: {ve}")
            sys.exit(1)

    def stop_simulation(self):
        traci.close()
        print("Simulation stopped.")

    def get_state(self, junction):
        lanes = traci.trafficlight.getControlledLanes(junction)
        state = []
        for lane in lanes:
            state.append(traci.lane.getLastStepVehicleNumber(lane))
        return state

    def get_reward(self, junction):
        lanes = traci.trafficlight.getControlledLanes(junction)
        waiting_time = 0
        for lane in lanes:
            waiting_time += traci.lane.getWaitingTime(lane)
        return -waiting_time

    def set_traffic_light(self, junction, phase, duration):
        traci.trafficlight.setRedYellowGreenState(junction, phase)
        traci.trafficlight.setPhaseDuration(junction, duration)


class DQNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RLAgent:
    def __init__(self, config, action_space):
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.batch_size = config['batch_size']
        self.memory = []
        self.memory_limit = config['memory_limit']
        self.action_space = action_space
        self.model = DQNModel(config['input_dim'], config['hidden_dim'], len(action_space), config['lr'])

    def store_experience(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        state_tensor = torch.tensor([state], dtype=torch.float).to(self.model.device)
        actions = self.model(state_tensor)
        return torch.argmax(actions).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(self.model.device)
        actions = torch.tensor(actions).to(self.model.device)
        rewards = torch.tensor(rewards).to(self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.model.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.model.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.model(next_states).max(1)[0]
        next_q_values[dones] = 0.0
        target_q_values = rewards + self.gamma * next_q_values

        loss = self.model.loss(target_q_values, q_values)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.model.device)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file '{path}' not found.")
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error loading model: {e}")
            sys.exit(1)


# **FIXED: train_agent function is now outside all classes**
def train_agent(config_path):
    try:
        config = load_config(config_path)

        print(f"Model save path: '{config.get('model_save_path', 'MISSING')}'")  
        print(f"Trip info output: '{config.get('tripinfo_output', 'MISSING')}'")

        # Check if paths are empty and exit with an error message
        if not config.get('model_save_path'):
            print("Error: 'model_save_path' is empty or missing in config file.")
            sys.exit(1)
        
        if not config.get('tripinfo_output'):
            print("Error: 'tripinfo_output' is empty or missing in config file.")
            sys.exit(1)
        
        ensure_dir(os.path.dirname(config['model_save_path']))
        ensure_dir(os.path.dirname(config['tripinfo_output']))


        env = SumoEnvironment(config)
        agent = RLAgent(config['agent'], [0, 1, 2, 3])

        env.start_simulation()

        for epoch in range(config['epochs']):
            print(f"Epoch {epoch + 1}/{config['epochs']}")
            state = env.get_state(env.junctions[0])
            total_reward = 0

            for step in range(config['steps']):
                action = agent.choose_action(state)
                env.set_traffic_light(env.junctions[0], config['phases'][action], config['duration'])
                reward = env.get_reward(env.junctions[0])
                next_state = env.get_state(env.junctions[0])
                done = step == config['steps'] - 1

                agent.store_experience(state, action, reward, next_state, done)
                agent.train()

                state = next_state
                total_reward += reward

                if done:
                    break

            print(f"Total Reward for Epoch {epoch + 1}: {total_reward}")

        agent.save_model(config['model_save_path'])
        env.stop_simulation()

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving progress...")
        agent.save_model(config['model_save_path'])
        env.stop_simulation()
        sys.exit(0)
    except Exception as e:
       print(f"Unexpected error: {e}")
       if 'env' in locals():
          env.stop_simulation()  
    sys.exit(1)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-c", dest="config", help="Path to config file", default="config.json")
    options, _ = parser.parse_args()

    train_agent(options.config)

