import time
import torch

from utilities.utils import write_file

class PolicyDeployer:
    def __init__(self, env, actor_model, device, control_dt, fname_test, ppo_steps):
        self.env = env
        self.actor_model = actor_model
        self.device = device
        self.control_dt = control_dt
        self.fname_test = fname_test
        self.ppo_steps = ppo_steps

    def run(self):
        voltage = self.env.sensors_in.measure_voltage_corrected()
        state = self.env.get_state(voltage)

        self.actor_model.eval()
        frame_idx = 0

        while True:
            t0 = time.perf_counter()

            with torch.inference_mode():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                dist = self.actor_model(state_tensor)
                action = dist.sample()
                prob_open = dist.probs.item()

            next_state, reward, next_voltage = self.env.step(bool(action.item()))


            state_val = state[0] if hasattr(state, "__len__") else state


            v = list(voltage)
            if len(v) != 6:
                raise ValueError(f"Expected 6 voltages, got {len(v)}")

            row = [
                frame_idx + 1,  # frame_idx
                int(action.item()),  # action
                float(state_val),  # state
                float(prob_open),  # prob_open
                float(reward),  # reward
                v[0], v[1], v[2], v[3], v[4], v[5],  # volt_ai0..5
            ]
            write_file(self.fname_test, row)

            state = next_state
            voltage = next_voltage
            frame_idx += 1

            if frame_idx >= int(self.ppo_steps):
                break

            while (time.perf_counter() - t0) < self.control_dt:
                pass