"""
This file contains PPO training algorithm
"""
import time
import threading #to make update and maintenance run concurrently
import copy
import torch
import torch.nn as nn
import numpy as np
from measurement.sensors import OutputManager
from utilities.utils import write_file

#construct PPOTraining class
class PPOTraining:
    def __init__(
        self,
        env,
        device,
        actor_model,
        critic_model,
        actor_optimizer,
        critic_optimizer,
        ppo_train_epochs,
        ppo_steps,
        ppo_update_epochs,
        reward_alpha,
        ppo_lower_return_limit,
        ppo_upper_return_limit,
        own_returns_flag,
        gamma,
        epsilon_clip,
        entropy_beta,
        save_interval,
        enable_maintenance,
        control_dt,
        reward_window,
        chkpnt_dir,
        fname_batch,
        fname_update,
        fname_ppo_epoch,
    ):
        """
        Initializes PPOTraining object.
        """
        self.env = env
        self.device = device

        self.actor_model = actor_model
        self.actor_optimizer = actor_optimizer
        self.critic_model = critic_model
        self.critic_optimizer = critic_optimizer

        self.ppo_train_epochs = ppo_train_epochs
        self.ppo_steps = ppo_steps
        self.ppo_update_epochs = ppo_update_epochs
        self.own_returns_flag=own_returns_flag
        self.alpha = reward_alpha
        self.ppo_lower_return_limit = ppo_lower_return_limit
        self.ppo_upper_return_limit = ppo_upper_return_limit
        self.reward_window=reward_window
        self.gamma = gamma

        self.epsilon_clip = epsilon_clip
        self.entropy_beta = entropy_beta
        self.save_interval = save_interval
        self.enable_maintenance = enable_maintenance
        self.chkpnt_dir = chkpnt_dir
        self.fname_batch = fname_batch
        self.fname_update = fname_update
        self.fname_epoch=fname_ppo_epoch
        self.control_dt = control_dt  # maintenance loop period in seconds
        self._maintenance_stop_event = None

    def compute_own_return(self, weighted_rewards):
        returns=[]
        assert self.ppo_upper_return_limit >= 1
        weighted_rewards=torch.cat(weighted_rewards, dim=0)  # [ppo_steps,1]
        for step in range(self.ppo_steps):
            end=min(self.ppo_steps, step + self.ppo_upper_return_limit)
            interval = weighted_rewards[step:end]
            discounts = (self.gamma ** torch.arange(interval.shape[0], device=weighted_rewards.device)).view(-1, 1)
            returns.append((interval * discounts).sum(dim=0, keepdim=True))
        return returns

    def compute_weighted_rewards(self, rewards):
        raw_rewards = torch.cat(rewards, dim=0)  # [ppo_steps,1]
        weighted_rewards = []
        for step in range(self.ppo_steps):
            end = min(self.ppo_steps, step + self.reward_window)
            avg = raw_rewards[step:end].sum()/(end-step)
            weighted_rewards.append(self.alpha * raw_rewards[step] + (1 - self.alpha) * avg.view(1, 1))
        return weighted_rewards

    def compute_mc_returns(self, rewards):
        returns=[]
        discounted_reward = 0
        for step in reversed(range(len(rewards))):
            discounted_reward = rewards[step]+self.gamma*discounted_reward
            returns.insert(0, discounted_reward)
        return returns

    # Maintenance actuation loop to be run in a separate thread, can be used for inference after training as well
    def maintenance_actuation_loop(self, deploy_actor_model, stop_event):
        voltage= self.env.sensors_in.measure_voltage_corrected()
        s = self.env.get_state(voltage)
        deploy_actor_model.eval()
        step_idx = 0  # for debugging purposes does not impede functionality
        while not stop_event.is_set():
            start_time = time.perf_counter()
            with torch.inference_mode():
                state_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device)
                dist = deploy_actor_model(state_tensor)
                action = dist.sample()  # same principle as in collect_batch
            next_state, _, _ = self.env.step(bool(action.item()))
            s = next_state
            while (time.perf_counter() - start_time) < self.control_dt:
                pass
            step_idx += 1


    def collect_batch(self):
        act_log_probs, values, states, actions, rewards, dist_probs, voltages, next_states = [], [], [], [], [], [], [], []
        state,voltage = self.env.reset()
        for step in range(self.ppo_steps):
            step_start_time = time.perf_counter() #read start time for pacing
            state_tensor = torch.FloatTensor(state).to(self.device)
            dist = self.actor_model(state_tensor)  # get distribution variable p from actor model
            value = self.critic_model(state_tensor).to(self.device)  # get predicted value from critic. value needs to be a tensor for the reward calculation
            action = dist.sample()

            next_state, reward, next_voltage_corrected = self.env.step(bool(action.item()))
            act_log_prob=dist.log_prob(action)
            act_log_probs.append(act_log_prob.unsqueeze(0))
            values.append(value.unsqueeze(0))
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward.flatten().float().to(self.device)
            else:
                reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
            rewards.append(reward_tensor.unsqueeze(0))  # shape [1, 1]
            states.append(state_tensor.unsqueeze(0))
            actions.append(action.unsqueeze(0))
            dist_probs.append(dist.probs.unsqueeze(0))
            next_states.append(torch.FloatTensor(next_state).to(self.device).unsqueeze(0))# (for stats)
            voltages.append(np.array(voltage))
            voltage = next_voltage_corrected
            state = next_state
            while (time.perf_counter() - step_start_time) < self.control_dt:
                pass
        return actions, states, next_states, rewards, values, dist_probs, act_log_probs, voltages
    @staticmethod
    def compute_selected_gamma(states):
        s = torch.cat(states, dim=0).reshape(-1)  # flatten tensor to treat it as list of states
        return float((s > 0).float().mean().item())

    #output of collect_batch in correct order: actions, states, next_states, rewards, values, dist_probs, act_log_probs, voltages
    def ppo_update(self, train_epoch, actions, states, act_log_probs, returns, advantages):
            """
            Function to update the policy and value network.
                1. pass state into networks, obtain predicted actions, values and new_log_probs
                2. calculate surrogate policy loss and mean squared error value loss
                3. backpropagate the losses through networks using Stochastic Gradient Descent (SGD)
            """
            epoch_numbers = []
            epoch_actor_losses = []
            epoch_critic_losses = []
            entropy_losses = []
            epoch_entropies = []
            clamped_fractions = []

            for update_epoch in range(1, self.ppo_update_epochs + 1):

                values = self.critic_model(states)

                dists = self.actor_model(states)

                entropy = dists.entropy().mean()
                new_log_probs = dists.log_prob(actions)

                ratio = (new_log_probs - act_log_probs).exp()

                surr1 = ratio * advantages
                surr2 = (torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages)

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss=nn.MSELoss(reduction="mean")(values,returns)

                entropy_loss = - self.entropy_beta * entropy

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                if train_epoch > 0:
                    self.actor_optimizer.zero_grad()
                    (actor_loss + entropy_loss).backward()
                    self.actor_optimizer.step()

                epoch_numbers.append(update_epoch)
                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                epoch_entropies.append(entropy.item())

                clamped = (ratio < (1.0 - self.epsilon_clip)) | (ratio > (1.0 + self.epsilon_clip))
                clamped_fraction = clamped.float().mean().item()
                clamped_fractions.append(clamped_fraction)

            return (epoch_numbers,
                epoch_actor_losses,
                epoch_critic_losses,
                entropy_losses,
                epoch_entropies,
                clamped_fractions)

    def train(self):
        frame_idx=0
        update_frame=0
        for train_epoch in range (1, self.ppo_train_epochs + 1):
            print(f"\n##### Training Epoch: {train_epoch} #####\n")
            (actions, states, next_states, rewards, values, dist_probs, act_log_probs, voltages)=self.collect_batch()
            maintenance_thread = None
            maintenance_stop_event = None
            try:
                if self.enable_maintenance:
                    deploy_actor_model = copy.deepcopy(self.actor_model)
                    maintenance_stop_event = threading.Event()
                    self._maintenance_stop_event = maintenance_stop_event

                    maintenance_thread = threading.Thread(
                        target=self.maintenance_actuation_loop,
                        args=(deploy_actor_model, maintenance_stop_event),
                        daemon=True,
                    )
                    maintenance_thread.start()
                else:
                    om: OutputManager = self.env.sensors_out
                    om.open_valve()

                weighted_rewards = self.compute_weighted_rewards(rewards)
                batch_gamma = self.compute_selected_gamma(states)

                if self.own_returns_flag: returns = torch.cat(self.compute_own_return(weighted_rewards)).detach()
                else: returns=torch.cat(self.compute_mc_returns(weighted_rewards)).detach()

                act_log_probs = torch.cat(act_log_probs).detach()
                values = torch.cat(values).detach()
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.cat(actions)
                dist_probs = torch.cat(dist_probs).detach()
                rewards = torch.cat(rewards).detach()
                weighted_rewards = torch.cat(weighted_rewards).detach()

                advantages = (returns - values).detach()
                epoch_total_reward = rewards.sum().item()

                count_valve_open = actions.sum().item()

                for step in range(self.ppo_steps):
                    frame_idx += 1
                    ppo_batch_data = (
                        [
                            frame_idx,
                            train_epoch,
                            step+1,
                            actions[step, 0].item(),
                            states[step, 0].item(),
                            rewards[step, 0].item(),
                            weighted_rewards[step, 0].item(),
                            next_states[step, 0].item(),
                            returns[step, 0].item(),
                            values[step, 0].item(),
                            advantages[step, 0].item(),
                            dist_probs[step, 0].item(),
                            act_log_probs[step, 0].item(),
                        ]
                        + voltages[step].tolist()
                    )
                    write_file(self.fname_batch, ppo_batch_data)
                (update_epoch,epoch_actor_losses,epoch_critic_losses,entropy_losses,epoch_entropies,clamped_fractions)= self.ppo_update(train_epoch,actions,states,act_log_probs,returns,advantages)

                for update in range(self.ppo_update_epochs):
                    update_frame+=1
                    ppo_update_data = [
                        update_frame,
                        update+1,
                        epoch_actor_losses[update],
                        epoch_critic_losses[update],
                        entropy_losses[update],
                        epoch_entropies[update],
                        clamped_fractions[update]
                    ]
                    write_file(self.fname_update, ppo_update_data)

                mean_state_ref = float(states.mean().item())
                mean_duty_cycle = float(actions.float().mean().item())
                gamma_all_steps = [float((vs > 0.0).sum()) / 6.0 for vs in voltages]
                mean_gamma_all_channels = float(np.mean(gamma_all_steps))
                ppo_epoch_data = [
                    int(train_epoch),
                    float(epoch_total_reward),
                    float(epoch_total_reward / self.ppo_steps),
                    float(np.mean(epoch_actor_losses)),
                    float(np.mean(epoch_critic_losses)),
                    float(np.mean(entropy_losses)),
                    mean_state_ref,
                    float(batch_gamma),
                    mean_gamma_all_channels,
                    mean_duty_cycle,
                ]
                write_file(self.fname_epoch, ppo_epoch_data)

                if train_epoch % self.save_interval == 0:
                    torch.save(
                        self.actor_model.state_dict(),
                        f"{self.chkpnt_dir}/epoch_{train_epoch}_torch_actor_model",
                    )
                    torch.save(
                        self.critic_model.state_dict(),
                        f"{self.chkpnt_dir}/epoch_{train_epoch}_torch_critic_model",
                    )
                    print("\n##### Models saved #####\n")
                print(
                    f"\nTraining Epoch: {train_epoch} \nTotal local Rewards of current Batch: {rewards.sum().item()} \nTotal weighted Rewards of current Batch: {weighted_rewards.sum().item()} \nMean Reward per step of current Batch: {rewards.sum().item() / self.ppo_steps}\nMean Weighted Reward per step of current Batch: {weighted_rewards.sum().item() / self.ppo_steps}\nMean Return of current Batch: {returns.sum().item() / self.ppo_steps} \nBatchsize: {self.ppo_steps}\nEpoch mean actor losses: {np.mean(epoch_actor_losses)}\nEpoch mean critic losses: {np.mean(epoch_critic_losses)}\nForward Flow Fraction at reference sensor (over batch): {batch_gamma}\nCount valve open: {count_valve_open}\nEntropy Losses: {np.mean(entropy_losses)}\nEntropy: {np.mean(epoch_entropies)}\n"
                )

            finally:
                if maintenance_stop_event is not None:
                    maintenance_stop_event.set()
                if maintenance_thread is not None:
                    maintenance_thread.join(timeout=2.0)