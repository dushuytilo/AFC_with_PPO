import datetime
from utils import mkdir
from utils import create_file
from utils import write_file


class LoggingUtils:
    """Utility class for logging setup and file management."""

    def __init__(self, PARAMS_FILE, BATCH_FILE, UPDATE_FILE, TEST_FILE, EPOCH_FILE, REWARD_ALPHA, EPSILON_CLIP, ENTROPY_BETA, GAMMA, PPO_UPPER_RETURN_LIMIT):
        """Initialize logging utilities."""

        self.PARAMS_FILE = PARAMS_FILE
        self.BATCH_FILE  = BATCH_FILE
        self.UPDATE_FILE = UPDATE_FILE
        self.TEST_FILE   = TEST_FILE
        self.EPOCH_FILE = EPOCH_FILE
        self.REWARD_ALPHA = REWARD_ALPHA
        self.EPSILON_CLIP = EPSILON_CLIP
        self.ENTROPY_BETA = ENTROPY_BETA
        self.GAMMA = GAMMA
        self.PPO_UPPER_RETURN_LIMIT = PPO_UPPER_RETURN_LIMIT
        return

    def create_log_dir(self):
        """Creates a log directory under logs/parameter_measurements/alpha_..._gamma_..._beta_..._epsilon_..."""

        def fmt(x):
            if isinstance(x, str):
                x = x.strip().replace(",", ".")
            try:
                x = float(x)
                s = f"{x:.12g}"
            except Exception:
                s = str(x)
            return s.replace(",", ".")
        combo = (
            f"alpha_{fmt(self.REWARD_ALPHA)}"
            f"_gamma_{fmt(self.GAMMA)}"
            f"_epsilon_{fmt(self.EPSILON_CLIP)}"
            f"_horizont_{fmt(self.PPO_UPPER_RETURN_LIMIT)}"
        )

        date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"logs/reproducibility_measurements/{date}_{combo}"
        self.log_dir = mkdir(".", base)
        self.models_dir = mkdir(".", f"{base}/model_checkpoints")

        return

    def create_log_ppo(self):
        """Creates log file for parameters"""

        fname_params = f"{self.log_dir}/{self.PARAMS_FILE}"
        create_file(
            fname_params,
            [
                "HIDDEN_SIZE",
                "ACTOR_LEARNING_RATE",
                "CRITIC_LEARNING_RATE",
                "GAMMA",
                "ALPHA_REWARD",
                "PPO_EPSILON_CLIP",
                "PPO_STEPS",
                "PPO_UPDATE_EPOCHS",
                "PPO_TRAIN_EPOCHS",
                "ENTROPY_BETA",
                "MAINTENANCE",
                "OWN_RETURNS",
                "UPPER RETURN BOUND"
            ],
        )

        return

    def create_log_ppo_epoch(self):
        """Creates log files for ppo epoch data"""
        self.fname_ppo_epoch = f"{self.log_dir}/{self.EPOCH_FILE}"
        create_file(
            self.fname_ppo_epoch,
            [
                "TRAIN_EPOCH",
                "TOTAL_REWARD",
                "MEAN_REWARD",
                "MEAN_ACTOR_LOSS",
                "MEAN_CRITIC_LOSS",
                "ENTROPY_LOSS",
                "MEAN_VOLTAGE_AT_REFERENCE",
                "MEAN_GAMMA_AT_REFERENCE",
                "MEAN_GAMMA_ALL_CHANNELS",
                "MEAN DUTY CYCLE",
            ],
        )
        return

    def create_log_ppo_batch(self, names_input_channel, state_sensors):
        """Creates log file for PPO batch data."""

        self.fname_batch = f"{self.log_dir}/{self.BATCH_FILE}"
        self.state_names = state_sensors
        self.state_names = [f"state_{sn}" for sn in self.state_names]
        self.names_input_channel = names_input_channel
        self.tau_input_names = [f"volt_{sn}" for sn in self.names_input_channel]
        self.gamma_state_names = [f"epoch_gamma_{sn}" for sn in self.names_input_channel]

        create_file(
            self.fname_batch,
            [
                "frame_idx",
                "train_epoch",
                "step",
                "action_t",
                "state_t",
                "local_reward_t+1",
                "weighted_reward",
                "state_t+1",
                "return",
                "value",
                "advantage",
                "dist_prob",
                "act_log_prob",
                "volt_ai0",
                "volt_ai1",
                "volt_ai2",
                "volt_ai3",
                "volt_ai4",
                "volt_ai5",
            ]
        )
        return

    def create_log_ppo_update(self):
        """Creates log file for PPO update data."""

        self.fname_update = f"{self.log_dir}/{self.UPDATE_FILE}"
        create_file(
            self.fname_update,
            [
                "frame_idx",
                "update_epoch",
                "actor_loss",
                "critic_loss",
                "update_entropy_losses",
                "update_entropies",
                "clamped_fraction",
            ],
        )

        return

    def create_params_file(self, HIDDEN_SIZE, ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE,
                           GAMMA, REWARD_ALPHA, EPSILON_CLIP, PPO_STEPS, PPO_UPDATE_EPOCHS, PPO_TRAIN_EPOCHS, ENTROPY_BETA, ENABLE_MAINTENANCE, OWN_RETURNS, PPO_UPPER_RETURN_LIMIT):
        """Writes the parameters to a file."""

        write_file(
            f"{self.log_dir}/{self.PARAMS_FILE}",
            [
                f"{HIDDEN_SIZE}",
                f"{ACTOR_LEARNING_RATE}",
                f"{CRITIC_LEARNING_RATE}",
                f"{GAMMA}",
                f"{REWARD_ALPHA}",
                f"{EPSILON_CLIP}",
                f"{PPO_STEPS}",
                f"{PPO_UPDATE_EPOCHS}",
                f"{PPO_TRAIN_EPOCHS}",
                f"{ENTROPY_BETA}",
                f"{ENABLE_MAINTENANCE}",
                f"{OWN_RETURNS}",
                f"{PPO_UPPER_RETURN_LIMIT}",
            ],
        )

        return

    def create_log_testdata(self):
        """Creates log file for test data."""
        create_file(
            f"{self.log_dir}/{self.TEST_FILE}",
            [
                "frame_idx",
                "action",
                "state",
                "prob_open",
                "reward",
                "volt_ai0",
                "volt_ai1",
                "volt_ai2",
                "volt_ai3",
                "volt_ai4",
                "volt_ai5",
            ]

        )
        return
