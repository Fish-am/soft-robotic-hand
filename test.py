import numpy as np
from stable_baselines3 import SAC
from env.hand_env import RoboticHandEnv

def main():
    # Create environment
    env = RoboticHandEnv(render_mode="human", task="fingertip_alignment")
    
    # Load the trained model
    model = SAC.load("models/robotic_hand_final")
    
    # Run evaluation episodes
    n_eval_episodes = 10
    success_count = 0
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{n_eval_episodes}")
        print(f"Target position: {env.target_pos}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Optional: slow down visualization
            import time
            time.sleep(0.01)
        
        print(f"Episode reward: {episode_reward}")
        if terminated:  # Successfully reached target
            success_count += 1
            print("Success!")
        else:
            print("Failed to reach target.")
    
    success_rate = success_count / n_eval_episodes
    print(f"\nOverall success rate: {success_rate * 100:.2f}%")

if __name__ == "__main__":
    main() 