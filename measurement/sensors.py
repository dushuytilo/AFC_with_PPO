"""
This file contains the Classes used for I/O and setup of the NI USB-6281 sensors.
"""

import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
import measurement.channel as channel
import measurement.calibration as calibration


class InputManager:
    """Class for I/O of the NI USB-6281, only for input channels"""

    def __init__(self, input_channel):
        """
        Initializes Sensors.

        Parameters
        ----------
        input_channel: list[str]
            list with strings of input channel names
        """

        ### Input channel name ###
        self.input_channel = ", ".join(input_channel)  # ai_channels need a comma separated str instead of a list

        ### Input task ###
        self.input_task = channel.ChannelVoltageIn(
            channel=self.input_channel,
            terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
            min_val=-10,
            max_val=10,
        )

        ### Sensor Calibration Class ###
        self.calib = calibration.Calibration()
    
    def measure_voltage(self):
        """Measures current wall shear stress on all mems sensors in Volt"""
        volt = np.array(self.input_task.read_single_voltage())
        return volt

    def measure_voltage_corrected(self):
        """Function for debugging"""
        volt = self.measure_voltage()
        corrections = [self.calib.offset_volt_channel[sensor] for sensor in self.calib.used_sensors]
        corr_volt = [volt[i] - corrections[i] for i in range(len(volt))]
        return corr_volt

    def close_tasks(self):
        """Closes all tasks. Call this function after all measurements are done"""
        self.input_task.close_task()


class OutputManager:
    """Class for I/O of the ethernet card NI 9181 with the NI 9254 module, only for output channels"""

    def __init__(self, output_channel):
        """
        Initializes Output Sensors.

        Parameters
        ----------
        output_channel: list[str]
            list with strings of the output channel names
        """

        ### Output channel names ###
        self.valve_channel = output_channel  # second output for pressure jet actuators ("Dev1/ao_i")

        ### Output tasks ###
        self.valve_output_task = channel.ChannelVoltageOut(channel=self.valve_channel, sampling_rate=1000, num_of_channels=len(self.valve_channel))

    def open_valve(self):
        """Start voltage output of 5 volt on valve channel (PJAs)"""
        self.valve_output_task.write_voltage_output(voltage=5)
        # print("valve opened")

    def close_valve(self):
        """Stop voltage output on valve channel (PJAs)"""
        self.valve_output_task.write_voltage_output(voltage=0.0)
        # print("valve closed")

    def close_tasks(self):
        """Closes all tasks. Call this function after all measurements are done"""
        self.valve_output_task.close_task()