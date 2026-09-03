import numpy as np
import time
from measurement.sensors import InputManager, OutputManager
from ppo.environment import WindTunnelEnv
import os

input_channels = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3", "Dev1/ai4", "Dev1/ai5"]
output_channels = ["Dev1/ao0"]

steps = 2000
step_duration = 0.006
pwm_period = 10
duration = steps * step_duration
# if you change pwm period please be sure to adjust the duty cycle inputs accordingly. 1 open step in 20 is 0.05 duty cycle etc. avoid having a float number of steps for open time
keys = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
choices = {"10": 1,
           "20": 2,
           "30": 3,
           "40": 4,
           "50": 5,
           "60": 6,
           "70": 7,
           "80": 8,
           "90": 9}

def run_offset_measurement():
    """Perform a single measurement to get the offset voltages."""
    # Confirm wind channel is OFF before proceeding, need stale state
    answer = input("Are you sure the wind channel is off? (yes/no): ").strip().lower()

    if answer in ("no", "n"):
        answer2 = input("Is it off now? (yes/no): ").strip().lower()
        if answer2 in ("no", "n"):
            raise RuntimeError("Please make sure the wind channel is turned off.")
        if answer2 not in ("yes", "y"):
            raise ValueError("Please answer with 'yes' or 'no'.")
    elif answer not in ("yes", "y"):
        raise ValueError("Please answer with 'yes' or 'no'.")

    print("Collecting offset voltages...")
    volt_log = [[] for _ in input_channels]
    om.close_valve()
    time.sleep(0.03)

    for step in range(steps):
        step_start_time = time.perf_counter()
        volt = im.measure_voltage()

        if len(volt) != len(volt_log):
            raise ValueError("Mismatch between sensor channels and voltage readings")

        for i, v in enumerate(volt):
            volt_log[i].append(v)
        # control step timing
        while (time.perf_counter() - step_start_time) < step_duration:
            pass

    print("Done collecting.")

    mean_offset_voltages = [np.mean(channel_data) for channel_data in volt_log]
    return mean_offset_voltages


def run_reference_measurement_no_actuation():

    print("Collecting real corrected voltages (PPO-style)...")
    volt_log = [[] for _ in input_channels]
    om.close_valve()
    time.sleep(10)

    for step in range(steps):
        step_start_time = time.perf_counter()
        corrected_volt = im.measure_voltage_corrected()
        if len(corrected_volt) != len(volt_log):
            raise ValueError("Mismatch between sensor channels and voltage readings")

        for i, v in enumerate(corrected_volt):
            volt_log[i].append(v)
        # control step timing
        while (time.perf_counter() - step_start_time) < step_duration:
            pass

    print("Done collecting.")
    om.open_valve()
    return volt_log


def run_reference_measurement_full_actuation():

    volt_log = [[] for _ in input_channels]
    print("Collecting real corrected voltages (PPO-style)...")
    om.open_valve()
    time.sleep(0.03)

    for step in range(steps):
        step_start_time = time.perf_counter()

        corrected_volt = im.measure_voltage_corrected()

        if len(corrected_volt) != len(volt_log):
            raise ValueError("Mismatch between sensor channels and voltage readings")

        for i, v in enumerate(corrected_volt):
            volt_log[i].append(v)
        # control step timing
        while (time.perf_counter() - step_start_time) < step_duration:
            pass

    om.close_valve()
    print("Done collecting.")

    return volt_log


def run_reference_measurement_ratio_actuation(duty_cycle_open):
    """Perform a single measurement with x ms/y ms actuation and return the corrected voltages.
    input is a fraction! not time in seconds. time in seconds is adjustable in the top variable step_duration
    pwm period is defined in steps not in seconds. time of pwm period is step_duration * pwm_period
    note that input period is defined in steps not time"""

    print("Collecting real corrected voltages ...")
    volt_log = [[] for _ in input_channels]
    open_steps = duty_cycle_open * pwm_period / (pwm_period)

    # initialization phase to reach steady state
    for step in range(2000):

        # Open and close the valve according to the duty cycle
        step_start_time = time.perf_counter()
        open_valve = (step % pwm_period) < open_steps
        if open_valve:
            om.open_valve()
        else:
            om.close_valve()
        while (time.perf_counter() - step_start_time) < step_duration:
            pass
    # measurement phase
    for step in range(steps):

        # Open and close the valve according to the duty cycle
        step_start_time = time.perf_counter()
        open_valve = (step % pwm_period) < open_steps
        if open_valve:
            om.open_valve()
        else:
            om.close_valve()
        corrected_volt = im.measure_voltage_corrected()
        for i, v in enumerate(corrected_volt): volt_log[i].append(v)
        while (time.perf_counter() - step_start_time) < step_duration:
            pass

    om.open_valve()
    print("Done collecting.")

    return volt_log

def calculate_forward_flow_fraction(voltages):
    # Calculate the forward flow fraction for a list of voltages.
    forward_flow_fraction = []
    for channel_data in voltages:
        pos_count = sum(1 for v in channel_data if v > 0)
        fff = pos_count / len(channel_data)
        forward_flow_fraction.append(fff)
    return forward_flow_fraction

def save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, duty_cycle_open, speed, mass):
    """Save mean voltages, corrected voltages to log files."""
    u_str = f"{speed}"
    m_str = f"{mass}"

    um_folder = f"_u_{u_str}_m_{m_str}"
    ref_folder = f"DC_{int(duty_cycle_open)}"
    base_dir = "logs/reference_measurements"
    log_folder = os.path.join(base_dir, um_folder, ref_folder)
    os.makedirs(log_folder, exist_ok=True)

    mean_log_filename = os.path.join(log_folder, "mean_voltages_log.txt")
    with open(mean_log_filename, "w") as f:
        f.write("Channel\tMean Corrected Voltage (V)\n")
        for ch, val in zip(input_channels, mean_voltages):
            f.write(f"{ch}\t{val:.6f}\n")
    print(f"Mean voltages saved to {mean_log_filename}")

    volt_log_filename = os.path.join(log_folder, "voltages_log.txt")
    with open(volt_log_filename, "w") as f:
        for line in volt_log:
            f.write(f"\t {line} \n")
    print(f"Voltages saved to {volt_log_filename}")

    fff_log_filename = os.path.join(log_folder, "FFF.txt")
    with open(fff_log_filename, "w") as f:
        f.write("Channel\tForward Flow Fraction\n")
        for ch, fff in zip(input_channels, forward_flow_fraction):
            f.write(f"{ch}\t{fff:.6f}\n")
    print(f"FFF saved to {fff_log_filename}")
    return

if __name__ == "__main__":
    try:
        print("Reference measurements\n")
        print("Choose Scenario:")
        print("0 - no actuation Dc 0")
        print("10 - actuation DC 0,1")
        print("20 - actuation DC 0,2")
        print("30 - actuation DC 0,3")
        print("40 - actuation DC 0,4")
        print("50 - actuation DC 0,5")
        print("60 - actuation DC 0,6")
        print("70 - actuation DC 0,7")
        print("80 - actuation DC 0,8")
        print("90 - actuation DC 0,9")
        print("100 - full actuation DC 1")
        print("1 - all of the above sequentially")
        print("2 - custom duty cycle actuation")
        choice = input("Select scenario: ")
        speed = input("Enter flow speed (m/s): ")
        mass = input("Enter mass flow rate (percent): ")

        env = WindTunnelEnv(output_channels, input_channels)
        im = env.sensors_in
        om = env.sensors_out

        # run selected scenario
        if choice == "0":
            volt_log = run_reference_measurement_no_actuation()
            mean_voltages = [np.mean(channel_data) for channel_data in volt_log]
            forward_flow_fraction = calculate_forward_flow_fraction(volt_log)
            save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, 0, speed, mass)
        # run all scenarios sequentially
        elif choice == "1":
            # run no actuation
            volt_log = run_reference_measurement_no_actuation()
            mean_voltages = [np.mean(channel_data) for channel_data in volt_log]
            forward_flow_fraction = calculate_forward_flow_fraction(volt_log)
            save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, 0, speed, mass)
            time.sleep(10)

            # run actuation over all duty cycles
            for key in keys:
                steps_open = choices[key]
                volt_log = run_reference_measurement_ratio_actuation(steps_open)
                mean_voltages = [np.mean(ch) for ch in volt_log]
                fff = calculate_forward_flow_fraction(volt_log)
                save_logs(input_channels, mean_voltages, volt_log, fff, key, speed, mass)
                time.sleep(10)

            # run full actuation
            volt_log = run_reference_measurement_full_actuation()
            mean_voltages = [np.mean(channel_data) for channel_data in volt_log]
            forward_flow_fraction = calculate_forward_flow_fraction(volt_log)
            save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, 100, speed, mass)

        # run single choices
        elif choice == "100":
            volt_log = run_reference_measurement_full_actuation()
            mean_voltages = [np.mean(channel_data) for channel_data in volt_log]
            forward_flow_fraction = calculate_forward_flow_fraction(volt_log)
            save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, 100, speed, mass)

        elif choice in keys:
            steps_open = choices[choice]
            volt_log = run_reference_measurement_ratio_actuation(steps_open)
            mean_voltages = [np.mean(ch) for ch in volt_log]
            forward_flow_fraction = calculate_forward_flow_fraction(volt_log)
            save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, steps_open, speed, mass)
        # run custom duty cycle
        elif choice == "2":
            steps_open = int(input("Open time in steps: "))
            volt_log = run_reference_measurement_ratio_actuation(steps_open)
            mean_voltages = [np.mean(ch) for ch in volt_log]
            forward_flow_fraction = calculate_forward_flow_fraction(volt_log)
            save_logs(input_channels, mean_voltages, volt_log, forward_flow_fraction, steps_open, speed, mass)
        # invalid choice
        else:
            print("Invalid choice. Exiting.")

    finally:
        im.close_tasks()
        om.open_valve()
        om.close_tasks()