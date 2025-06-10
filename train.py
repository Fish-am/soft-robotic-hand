import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from env.hand_env import SoftRoboticHandEnv
import pybullet as p

class MaterialDomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        # Define ranges for material properties
        self.youngs_modulus_range = (8e3, 1.2e4)  # Range for Young's modulus
        self.poisson_ratio_range = (0.25, 0.35)    # Range for Poisson ratio
        self.density_range = (900, 1100)           # Range for material density
        self.damping_range = (3.0, 7.0)            # Range for damping
        self.friction_range = (0.8, 1.2)           # Range for friction
        
    def reset(self, **kwargs):
        # Randomize material properties
        self._randomize_material_properties()
        return self.env.reset(**kwargs)
    
    def _randomize_material_properties(self):
        # Randomize FEM and material properties for each finger
        for i in range(self.env.num_fingers):
            # Sample random material properties
            youngs_modulus = np.random.uniform(*self.youngs_modulus_range)
            poisson_ratio = np.random.uniform(*self.poisson_ratio_range)
            density = np.random.uniform(*self.density_range)
            damping = np.random.uniform(*self.damping_range)
            friction = np.random.uniform(*self.friction_range)
            
            # Update material properties in PyBullet
            p.changeDynamics(
                self.env.robot_id,
                i,
                mass=density * 0.001,  # Scale mass based on density
                lateralFriction=friction,
                spinningFriction=friction * 0.1,
                rollingFriction=friction * 0.1,
                restitution=0.3,
                contactStiffness=youngs_modulus * 0.1,
                contactDamping=damping,
                linearDamping=damping * 0.1,
                angularDamping=damping * 0.1
            )
            
            # Update environment parameters
            self.env.youngs_modulus = youngs_modulus
            self.env.poisson_ratio = poisson_ratio
            self.env.mass_density = density
            self.env.damping = damping
            self.env.friction_coeff = friction

def main():
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create environment with domain randomization
    env = SoftRoboticHandEnv(render_mode="direct", task="fingertip_alignment")
    env = MaterialDomainRandomizationWrapper(env)
    
    # Add action noise for exploration
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )
    
    # Create the SAC agent with custom parameters optimized for soft bodies
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy=None,
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256, 128],  # Policy network
                "qf": [256, 256, 128]   # Q-function network
            },
            "activation_fn": "relu",
            "log_std_init": -3.0  # Lower initial exploration
        },
        tensorboard_log="./logs/"
    )

    # Create checkpoint callback with more frequent saves for FEM stability
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save more frequently
        save_path="./models/",
        name_prefix="soft_hand_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    # Train the agent with increased timesteps for better FEM convergence
    total_timesteps = 2000000  # Increased training time
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name="FEM_training"
    )

    # Save the final model
    model.save("models/soft_hand_final")

if __name__ == "__main__":
    main() 