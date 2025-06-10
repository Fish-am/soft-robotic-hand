import numpy as np
import matplotlib.pyplot as plt
from env.hand_env import SoftRoboticHandEnv

def measure_pressure_response(real_pressure_data, real_deformation_data):
    """
    Analyze real robot data to extract pressure-to-force relationship
    real_pressure_data: List of pressure readings from real hand
    real_deformation_data: List of corresponding deformation measurements
    """
    # Fit polynomial to pressure-deformation relationship
    coeffs = np.polyfit(real_pressure_data, real_deformation_data, 3)
    
    # Calculate pressure to force ratio
    pressure_to_force = np.mean(np.gradient(real_deformation_data) / 
                              np.gradient(real_pressure_data))
    
    # Calculate response time (time to reach 63.2% of final value)
    response_times = []
    for i in range(len(real_pressure_data)-1):
        if abs(real_pressure_data[i+1] - real_pressure_data[i]) > 10:  # Step change
            final_value = real_pressure_data[i+1]
            start_value = real_pressure_data[i]
            target = start_value + 0.632 * (final_value - start_value)
            
            # Find time to reach target
            for j in range(i, len(real_pressure_data)):
                if abs(real_pressure_data[j] - target) < 1.0:
                    response_times.append(j - i)
                    break
    
    response_time = np.mean(response_times) * 0.01  # Convert steps to seconds
    
    return {
        'pressure_to_force_ratio': pressure_to_force,
        'response_time': response_time,
        'pressure_coeffs': coeffs
    }

def measure_noise_characteristics(real_pressure_data, real_position_data):
    """
    Analyze sensor noise from real robot data
    """
    # Calculate pressure noise
    pressure_noise = np.std(np.diff(real_pressure_data))
    
    # Calculate position noise
    position_noise = np.std(np.diff(real_position_data))
    
    return {
        'pressure_noise_std': pressure_noise,
        'position_noise_std': position_noise
    }

def measure_valve_characteristics(real_pressure_data, commanded_pressures):
    """
    Analyze valve behavior from real robot data
    """
    # Find minimum pressure change that produces movement
    pressure_diffs = np.abs(np.diff(real_pressure_data))
    valid_changes = pressure_diffs[pressure_diffs > 0]
    deadband = np.percentile(valid_changes, 10)  # Use 10th percentile as deadband
    
    # Find maximum deformation
    max_deform = np.max(np.abs(real_pressure_data))
    
    return {
        'valve_deadband': deadband,
        'max_finger_deformation': max_deform
    }

def calibrate_simulation(real_data_file):
    """
    Main calibration function
    real_data_file: Path to file containing real robot data
    """
    # Load real robot data
    # This should be replaced with your actual data loading code
    real_data = np.load(real_data_file)
    real_pressure_data = real_data['pressures']
    real_deformation_data = real_data['deformations']
    real_position_data = real_data['positions']
    commanded_pressures = real_data['commands']
    
    # Measure various characteristics
    pressure_response = measure_pressure_response(
        real_pressure_data, 
        real_deformation_data
    )
    
    noise_chars = measure_noise_characteristics(
        real_pressure_data,
        real_position_data
    )
    
    valve_chars = measure_valve_characteristics(
        real_pressure_data,
        commanded_pressures
    )
    
    # Combine all parameters
    calibration_params = {
        'pressure_to_force_ratio': pressure_response['pressure_to_force_ratio'],
        'finger_response_time': pressure_response['response_time'],
        'pressure_noise_std': noise_chars['pressure_noise_std'],
        'position_noise_std': noise_chars['position_noise_std'],
        'valve_deadband': valve_chars['valve_deadband'],
        'max_finger_deformation': valve_chars['max_finger_deformation']
    }
    
    # Validate calibration
    env = SoftRoboticHandEnv(
        render_mode="human",
        calibration_params=calibration_params
    )
    
    # Run validation test
    obs = env.reset()[0]
    total_error = 0
    n_steps = min(len(real_pressure_data), 1000)
    
    for i in range(n_steps):
        action = commanded_pressures[i]
        obs, _, _, _ = env.step(action)
        
        # Compare with real data
        sim_pressure = obs[:5]  # First 5 values are pressures
        real_pressure = real_pressure_data[i]
        error = np.mean(np.abs(sim_pressure - real_pressure))
        total_error += error
    
    avg_error = total_error / n_steps
    print(f"Average simulation error: {avg_error:.4f} PSI")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(real_pressure_data[:n_steps], label='Real')
    plt.plot([obs[:5] for obs in env.pressure_history], label='Simulation')
    plt.title('Pressure Response Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Pressure (PSI)')
    plt.legend()
    plt.savefig('calibration_results.png')
    
    return calibration_params

if __name__ == "__main__":
    # Replace with path to your real robot data
    calibration_params = calibrate_simulation('real_robot_data.npy')
    print("\nCalibrated Parameters:")
    for key, value in calibration_params.items():
        print(f"{key}: {value}") 