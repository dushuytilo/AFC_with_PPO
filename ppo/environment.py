"""
This file contains the wind tunnel environment defined for PPO.
"""

import time
import numpy as np
import measurement.sensors as sm
from measurement.sensors import InputManager, OutputManager
from threading import Thread


class WindTunnelEnv:
    """Class for custom wind tunnel environment"""

    def __init__(self, output_channel, input_channel, selected_states):
        """Initialize wind tunnel environment"""
        self.output_channel  = output_channel
        self.input_channel   = input_channel
        self.selected_states = selected_states
        self.sensors_in  = sm.InputManager(input_channel=self.input_channel)
        self.sensors_out = sm.OutputManager(output_channel=self.output_channel)

    def get_state(self, corrected_voltages):
        v = np.asarray(corrected_voltages, dtype=np.float32).reshape(-1)
        idx = np.asarray(self.selected_states, dtype=int)
        return v[idx]

    @staticmethod
    def calc_reward(state):
        """Helper function to turn measured voltage into reward"""

        conditions_for_reward = [state < 0.0, state >= 0.0]
        values_for_reward = [-1, 1]

        reward_array = np.select(conditions_for_reward, values_for_reward)
        reward = np.array(reward_array.sum()).reshape(1)

        return reward

    def reset(self):
        """new episode"""
        measured_volt = np.array(self.sensors_in.measure_voltage_corrected())
        state = self.get_state(measured_volt)

        return state, measured_volt

    def step(self, action):
        """Computes the state of the environment after applying an action"""

        if action:
            self.sensors_out.open_valve()
        else:
            self.sensors_out.close_valve()

        measure_voltage_corrected = self.sensors_in.measure_voltage_corrected()
        state = self.get_state(measure_voltage_corrected)
        reward = self.calc_reward(state)

        return state, reward, measure_voltage_corrected

    def end_measurement(self):
        self.sensors_out.open_valve()
        self.sensors_out.close_tasks()
        self.sensors_in.close_tasks()