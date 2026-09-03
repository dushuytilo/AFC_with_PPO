"""
This file contains the main entry point of the project where parameters can be set.
"""
import time
import os
import psutil
import torch
from torch import optim

from ppo.deploy import PolicyDeployer
from ppo.network import ActorNetwork
from ppo.network import CriticNetwork
from ppo.environment import WindTunnelEnv
from ppo.training import PPOTraining

import utilities.logging_utils as lu

# Parameter Configuration           # default | description
HIDDEN_SIZE             = 64        # = 64    | Number of neurons in hidden layer in NN for number of Hidden layers refer to ,/ppo/networks.py current set-up uses 2 hidden layers
ACTOR_LEARNING_RATE     = 1e-4      # = 1e-4  | Standard for Actor network adam optimizer with tighter epsilon-clip
CRITIC_LEARNING_RATE    = 3e-4      # = 3e-4  | Standard for Critic network adam optimizer
REWARD_ALPHA            = 0.3       # = 0.1, 0.3, 0.5  | Weighting factor for the two reward components (0 = only mean reward over interval (global), 1 = only local reward, local reward already considers temporal difference
GAMMA                   = 0.99      # = 1.00  | Discount factor to calculate returns
ENTROPY_BETA            = 0.0       # = 0.01  | Beta value for entropy loss (higher = more exploration, lower = more exploitation) look at the possibility to use entropy bonus to counter randomness of environment.
EPSILON_CLIP            = 0.1       # = 0.2   | Used to clip the ratio between new and old policy. 0.2 bei Font 2025
PPO_STEPS               = 1000      # = 1000  | Number of transitions sampled for each training iteration = batch_size
PPO_UPDATE_EPOCHS       = 100       # = 100   | Number of pass over entire batch of training data number of policy updates
PPO_TRAIN_EPOCHS        = 100       # = 2000  | Limit epochs for the PPO training loop
PPO_UPPER_RETURN_LIMIT  = 100       # = 4     | Upper limit for the reward
PPO_LOWER_RETURN_LIMIT  = 0         # = 1     | Lower limit for the reward. 0 because it considers rewards already adjusted to temporal difference
REWARD_WINDOW           = 5         # Size of moving average window for weighted reward calculation
SAVE_INTERVAL           = 20        # = 50    | interval for model checkpoints
CONTROL_DT              = 0.006     # Time interval of one step in seconds, measured through profiling, should be adjusted to individual hardware setups

LOAD_MODEL              = False     # Select to load an existing torch model
ENABLE_MAINTENANCE      = True      # Set to True to enable maintenance mode inbetween rollouts (valve open otherwise)
OWN_RETURNS             = True      # If True, use own return calculation instead of that from PPO paper
DEPLOY                  = False     # Set to True for deployment (no training, only inference)

PARAMS_FILE             = "parameter.txt"
BATCH_FILE              = "batch_data.txt"
UPDATE_FILE             = "update_data.txt"
TEST_FILE               = "model_test.txt"
EPOCH_FILE              = "ppo_epoch_data.txt"

MODEL_LOAD_PATH_ACTOR = ("./PATH")
MODEL_LOAD_PATH_CRITIC = ("./PATH")

# NI USB-6281 I/O channel names
names_input_channel  = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5"]
state_sensors = [3] #reference sensor ai3 (so 4th sensor, 0-indexed list)

names_output_channel = ["Dev1/ao0"] #all valves are controlled via ao0 in-phase
num_actions = 1

if __name__ == "__main__":

    pid = os.getpid()
    process = psutil.Process(pid)
    print(f"\nProcess ID: {pid}\n")

    if os.name == "nt":  # Windows
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Increased priority of Process")
    # else:  # Linux/Mac (needs root privilege)
        # process.nice(-10)
    
    # Logging Configuration
    Logger = lu.LoggingUtils(PARAMS_FILE, BATCH_FILE, UPDATE_FILE, TEST_FILE, EPOCH_FILE, REWARD_ALPHA, EPSILON_CLIP, ENTROPY_BETA, GAMMA, PPO_UPPER_RETURN_LIMIT)
    Logger.create_log_dir()
    Logger.create_log_ppo()
    Logger.create_log_ppo_epoch()
    Logger.create_log_ppo_batch(names_input_channel, state_sensors)
    Logger.create_log_ppo_update()
    Logger.create_params_file(HIDDEN_SIZE, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE,
                              GAMMA, REWARD_ALPHA, EPSILON_CLIP, PPO_STEPS, PPO_UPDATE_EPOCHS,
                              PPO_TRAIN_EPOCHS, ENTROPY_BETA, ENABLE_MAINTENANCE, OWN_RETURNS, PPO_UPPER_RETURN_LIMIT)

    # Select CUDA if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"\nDevice: {device}\n")

    # Initialize environment
    env = WindTunnelEnv(output_channel=names_output_channel, input_channel=names_input_channel, selected_states=state_sensors)
    im = env.sensors_in
    om = env.sensors_out
    # Initialize Networks
    actor_model = ActorNetwork(num_inputs=len(state_sensors), num_outputs=num_actions, hidden_size=HIDDEN_SIZE).to(device)
    print(f"Actor Model:\n\n{actor_model}\n")
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=ACTOR_LEARNING_RATE)

    critic_model = CriticNetwork(num_inputs=len(state_sensors), hidden_size=HIDDEN_SIZE).to(device)
    print(f"Critic Model:\n\n{critic_model}\n")
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=CRITIC_LEARNING_RATE)

    # Load model if applicable
    if LOAD_MODEL:
        actor_model.load_state_dict(torch.load(MODEL_LOAD_PATH_ACTOR))
        critic_model.load_state_dict(torch.load(MODEL_LOAD_PATH_CRITIC))
        print("\n##### Models loaded #####\n")

    # Start PPO training ###
    if DEPLOY:
        Logger.create_log_testdata()
        fname_test = f"{Logger.log_dir}/{TEST_FILE}"
        print("\n\n##### Deployment Mode: Inference Only #####\n")

        # load actor weights before running
        actor_model.load_state_dict(torch.load(MODEL_LOAD_PATH_ACTOR))
        actor_model.eval()
        print("\n##### Models loaded #####\n")
        start_time = time.perf_counter()

        deployer = PolicyDeployer(
            env=env,
            actor_model=actor_model,
            device=device,
            control_dt=CONTROL_DT,
            fname_test=fname_test,
            ppo_steps=PPO_STEPS,
        )
        deployer.run()

        end_time = time.perf_counter()
        duration = (end_time - start_time) / 60.0

        print("\n##### End Of Deployment #####\n\n")
        print(f"\nDuration in Minutes: {duration}\n")
        print("\n##### Closing Tasks #####\n\n")
        im.close_tasks()
        om.open_valve()
        om.close_tasks()

        exit()

    else:
        policy_trainer = PPOTraining(
            env=env,
            device=device,
            actor_model=actor_model,
            critic_model=critic_model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            ppo_train_epochs=PPO_TRAIN_EPOCHS,
            ppo_steps=PPO_STEPS,
            ppo_update_epochs=PPO_UPDATE_EPOCHS,
            reward_alpha=REWARD_ALPHA,
            own_returns_flag=OWN_RETURNS,
            ppo_upper_return_limit=PPO_UPPER_RETURN_LIMIT,
            ppo_lower_return_limit=PPO_LOWER_RETURN_LIMIT,
            gamma=GAMMA,
            epsilon_clip=EPSILON_CLIP,
            entropy_beta=ENTROPY_BETA,
            save_interval=SAVE_INTERVAL,
            enable_maintenance=ENABLE_MAINTENANCE,
            control_dt=CONTROL_DT,
            reward_window=REWARD_WINDOW,
            chkpnt_dir=Logger.models_dir,
            fname_batch=Logger.fname_batch,
            fname_update=Logger.fname_update,
            fname_ppo_epoch=Logger.fname_ppo_epoch,
        )

        print("\n\n##### Start Of Training #####\n")
        start_time=time.perf_counter()
        policy_trainer.train()
        end_time=time.perf_counter()
        duration=(end_time-start_time)/60.0
        print(f"\nDuration in Minutes: {duration}\n")
        print("\n##### End Of Training #####\n\n")

        im.close_tasks()
        om.open_valve()
        om.close_tasks()

