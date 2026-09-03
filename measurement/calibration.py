"""
This file contains a class to handle the calibration of the mems sensors.

To get an correct wall sheer stress each voltage which is measured on the mems
sensors gets subtracted by the voltage offset on the wind channel setup.
Than the corrected voltage is passed to function
which maps an measured voltage to an wall sheer stress.
(This is done in sensors)
"""

class Calibration:
    def __init__(self):
        self.used_sensors = [0, 1, 2, 3, 4, 5]

        self.offset_volt_channel = {
            0: 0.000003,
            1: 0.060675,
            2: 0.036344,
            3: 0.002520,
            4: 0.001241,
            5: 0.002680
        }
