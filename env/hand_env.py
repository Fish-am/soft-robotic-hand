import os
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

class SoftRoboticHandEnv(gym.Env):
    def __init__(self, render_mode="human", task="fingertip_alignment", calibration_params=None):
        super().__init__()
        
        # Load calibration parameters or use defaults
        self.calibration_params = calibration_params or {
            # These should be measured from your real hand
            'pressure_to_force_ratio': 0.1,  # N/PSI
            'max_finger_deformation': 0.05,  # meters
            'finger_response_time': 0.1,     # seconds
            'valve_deadband': 5.0,           # PSI
            'pressure_noise_std': 2.0,       # PSI
            'position_noise_std': 0.002,     # meters
        }
        
        # PyBullet setup with FEM support
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Enable FEM soft body simulation
        p.setPhysicsEngineParameter(numSolverIterations=50)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setPhysicsEngineParameter(contactERP=0.8)
        p.setPhysicsEngineParameter(frictionERP=0.8)
        
        # Load ground plane
        p.loadURDF("plane.urdf")
        
        # Task selection
        self.task = task
        
        # Pneumatic parameters (based on SoMoGym paper)
        self.max_pressure = 100.0  # Maximum pressure in PSI
        self.min_pressure = 0.0    # Minimum pressure in PSI
        self.pressure_threshold = 60.0  # Safety threshold
        self.pressure_rate_limit = 20.0  # PSI/s rate limit for smooth actuation
        
        # FEM parameters for soft body simulation
        self.youngs_modulus = 1e4  # Young's modulus for soft material
        self.poisson_ratio = 0.3   # Poisson ratio for rubber-like material
        self.mass_density = 1000.0  # kg/m^3
        self.friction_coeff = 1.0
        
        # Soft body parameters (from SoMoGym)
        self.damping = 5.0  # Joint damping for soft behavior
        self.contact_stiffness = 500.0  # Contact stiffness
        self.contact_damping = 50.0  # Contact damping
        
        # Finger parameters
        self.num_fingers = 5  # thumb, index, middle, ring, little
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'little']
        
        # Current pressure state for each finger
        self.finger_pressures = np.zeros(self.num_fingers)
        
        # Get the absolute path to the URDF file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_dir = os.path.dirname(current_dir)
        self.urdf_path = os.path.join(workspace_dir, "urdf", "soft_hand.urdf")
        
        # Define action and observation spaces
        # Actions: pressure change for each finger [-1, 1] scaled to pressure range
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_fingers,),
            dtype=np.float32
        )
        
        # Observations: current pressures, finger positions, velocities, deformations, target position
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_fingers * 6 + 3,),  # pressures + positions + velocities + deformations + target
            dtype=np.float32
        )
        
        # Initialize robot state
        self.robot_id = None
        self.target_pos = None
        self.reset()
        
        # Add real-world constraints
        self.pressure_deadband = self.calibration_params['valve_deadband']
        self.response_delay_steps = int(self.calibration_params['finger_response_time'] / 0.01)  # Convert to timesteps
        self.pressure_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Load robot hand URDF with FEM properties
        self.robot_id = p.loadURDF(
            self.urdf_path, 
            [0, 0, 1], 
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Reset pressures
        self.finger_pressures = np.zeros(self.num_fingers)
        
        # Reset finger positions and apply soft body parameters
        for i in range(self.num_fingers):
            p.resetJointState(self.robot_id, i, 0)
            
            # Add FEM and soft body properties
            p.changeDynamics(
                self.robot_id, 
                i,
                jointDamping=self.damping,
                contactStiffness=self.contact_stiffness,
                contactDamping=self.contact_damping,
                lateralFriction=self.friction_coeff,
                spinningFriction=0.1,
                rollingFriction=0.1,
                restitution=0.3,
                linearDamping=0.1,
                angularDamping=0.1,
                maxJointVelocity=10,
                collisionMargin=0.001
            )
            
            # Set material properties for FEM
            p.setCollisionFilterGroupMask(self.robot_id, i, 0x1, 0x1)
            p.changeDynamics(
                self.robot_id,
                i,
                activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING,
                linearDamping=0.1,
                angularDamping=0.1,
                maxJointVelocity=10,
                contactProcessingThreshold=0.001,
                frictionAnchor=1,
                collisionMargin=0.001
            )
        
        # Set random target position
        self.target_pos = self._generate_random_target()
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        # Apply pressure changes (actions) to each finger with rate limiting
        dt = 0.01  # Time step
        for i in range(self.num_fingers):
            # Convert normalized action to pressure change
            desired_pressure_change = np.interp(
                action[i], 
                [-1, 1], 
                [-self.max_pressure/10, self.max_pressure/10]
            )
            
            # Apply rate limiting
            max_change = self.pressure_rate_limit * dt
            pressure_change = np.clip(
                desired_pressure_change,
                -max_change,
                max_change
            )
            
            # Update finger pressure
            new_pressure = np.clip(
                self.finger_pressures[i] + pressure_change,
                self.min_pressure,
                self.max_pressure
            )
            
            # Convert pressure to force using nonlinear FEM model
            force = self._pressure_to_force(new_pressure)
            
            # Apply force with FEM-based deformation
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.TORQUE_CONTROL,
                force=force,
                maxVelocity=1.0
            )
            
            # Apply additional FEM forces for soft body behavior
            self._apply_fem_forces(i, force)
            
            self.finger_pressures[i] = new_pressure
        
        # Simulate physics with smaller timesteps for FEM stability
        for _ in range(20):
            p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False
        
        # Add detailed info including deformation metrics
        info = {
            'pressures': self.finger_pressures.copy(),
            'over_pressure': np.any(self.finger_pressures > self.pressure_threshold),
            'finger_positions': self._get_finger_positions(),
            'finger_velocities': self._get_finger_velocities(),
            'deformations': self._get_deformations()
        }
        
        return observation, reward, terminated, truncated, info

    def _apply_fem_forces(self, finger_idx, base_force):
        """Apply FEM-based forces with real-world constraints"""
        # Get current finger state
        state = p.getLinkState(self.robot_id, finger_idx, computeLinkVelocity=1)
        pos = state[0]
        vel = state[6]
        
        # Add position noise to simulate sensor noise
        pos = np.array(pos) + np.random.normal(0, self.calibration_params['position_noise_std'], 3)
        
        # Calculate deformation with max limit from real hand
        rest_pos = p.getLinkState(self.robot_id, finger_idx)[2]
        deformation = np.array(pos) - np.array(rest_pos)
        deform_magnitude = np.linalg.norm(deformation)
        
        # Limit deformation to real-world maximum
        if deform_magnitude > self.calibration_params['max_finger_deformation']:
            deformation = deformation * (self.calibration_params['max_finger_deformation'] / deform_magnitude)
        
        # Calculate forces with calibrated parameters
        k_linear = self.youngs_modulus * self.calibration_params['pressure_to_force_ratio']
        k_nonlinear = k_linear * 0.1
        elastic_force = -(k_linear * deformation + k_nonlinear * deformation * deform_magnitude)
        
        # Add real-world damping
        damping_force = -self.contact_damping * np.array(vel)
        
        # Simulate actuation delay
        self.pressure_history.append(base_force)
        if len(self.pressure_history) > self.response_delay_steps:
            delayed_force = self.pressure_history.pop(0)
        else:
            delayed_force = 0
        
        # Combine forces with delayed actuation
        total_force = delayed_force * np.array([1, 1, 1]) + elastic_force + damping_force
        
        # Apply the combined force
        p.applyExternalForce(
            self.robot_id,
            finger_idx,
            total_force,
            pos,
            p.WORLD_FRAME
        )

    def _get_deformations(self):
        """Get current deformation state of each finger"""
        deformations = []
        for i in range(self.num_fingers):
            current_pos = p.getLinkState(self.robot_id, i)[0]
            rest_pos = p.getLinkState(self.robot_id, i)[2]
            deformation = np.array(current_pos) - np.array(rest_pos)
            deformations.append(deformation)
        return np.array(deformations)

    def _get_observation(self):
        obs = []
        
        # Add normalized pressure readings
        obs.extend(self.finger_pressures / self.max_pressure)
        
        # Add finger positions
        finger_positions = self._get_finger_positions()
        obs.extend(finger_positions.flatten())
        
        # Add finger velocities 
        finger_velocities = self._get_finger_velocities()
        obs.extend(finger_velocities.flatten())
        
        # Add deformation states
        deformations = self._get_deformations()
        obs.extend(deformations.flatten())
        
        # Add target position
        obs.extend(self.target_pos)
        
        return np.array(obs)

    def _pressure_to_force(self, pressure):
        # Add deadband and noise to better match real valves
        if abs(pressure) < self.pressure_deadband:
            pressure = 0
            
        # Add noise to simulate real-world pressure variations
        pressure += np.random.normal(0, self.calibration_params['pressure_noise_std'])
        
        # Convert using calibrated ratio and nonlinear terms
        k1 = self.calibration_params['pressure_to_force_ratio']
        k2 = k1 * 0.1  # Quadratic term
        k3 = k1 * 0.01  # Cubic term for hyperelastic behavior
        return k1 * pressure + k2 * pressure**2 + k3 * pressure**3

    def _get_finger_positions(self):
        positions = []
        for i in range(self.num_fingers):
            state = p.getLinkState(self.robot_id, i)
            positions.append(state[0])  # Position of finger tip
        return np.array(positions)

    def _get_finger_velocities(self):
        velocities = []
        for i in range(self.num_fingers):
            state = p.getLinkState(self.robot_id, i, computeLinkVelocity=1)
            velocities.append(state[6])  # Linear velocity of finger tip
        return np.array(velocities)

    def _compute_reward(self):
        if self.task == "fingertip_alignment":
            # Get active fingertip positions
            active_fingers = self._get_active_fingers()
            total_reward = 0
            
            for finger_idx in active_fingers:
                fingertip_pos = p.getLinkState(self.robot_id, finger_idx)[0]
                distance = np.linalg.norm(np.array(fingertip_pos) - np.array(self.target_pos))
                
                # Reward is negative distance (closer is better)
                finger_reward = -distance
                
                # Add bonus for very close alignment
                if distance < 0.05:
                    finger_reward += 10.0
                
                # Penalty for excessive pressure
                if self.finger_pressures[finger_idx] > self.pressure_threshold:
                    finger_reward -= 5.0
                
                # Add penalty for rapid pressure changes
                if finger_idx > 0:
                    pressure_change = abs(self.finger_pressures[finger_idx] - self.finger_pressures[finger_idx-1])
                    if pressure_change > self.pressure_rate_limit:
                        finger_reward -= pressure_change / self.pressure_rate_limit
                
                # Add penalty for excessive deformation
                deformation = np.linalg.norm(self._get_deformations()[finger_idx])
                if deformation > 0.05:  # Threshold for excessive deformation
                    finger_reward -= deformation * 10.0
                
                total_reward += finger_reward
            
            return total_reward / len(active_fingers)
        
        return 0.0

    def _is_done(self):
        if self.task == "fingertip_alignment":
            active_fingers = self._get_active_fingers()
            all_fingers_reached = True
            
            for finger_idx in active_fingers:
                fingertip_pos = p.getLinkState(self.robot_id, finger_idx)[0]
                distance = np.linalg.norm(np.array(fingertip_pos) - np.array(self.target_pos))
                
                # Check both position and deformation constraints
                deformation = np.linalg.norm(self._get_deformations()[finger_idx])
                if distance >= 0.02 or deformation > 0.05:  # Success thresholds
                    all_fingers_reached = False
                    break
            
            return all_fingers_reached
        
        return False

    def _get_active_fingers(self):
        if self.task == "fingertip_alignment":
            if "five" in self.task:
                return range(5)  # All fingers
            elif "three" in self.task:
                return [0, 1, 2]  # Thumb, index, middle
            elif "two" in self.task:
                return [0, 1]  # Thumb, index
        return [0]  # Default to thumb only

    def _generate_random_target(self):
        # Generate random target position within reasonable workspace
        x = np.random.uniform(-0.3, 0.3)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0.1, 0.5)
        return [x, y, z]

    def render(self):
        if self.render_mode == "human":
            # PyBullet GUI already handles rendering
            pass
        
    def close(self):
        p.disconnect(self.client)

    def get_real_world_action(self, sim_action):
        """Convert simulation action to real-world valve commands"""
        # Scale actions to real pressure range
        real_action = np.clip(sim_action, -1, 1) * self.max_pressure
        
        # Apply deadband
        real_action[np.abs(real_action) < self.pressure_deadband] = 0
        
        # Add noise to match real-world behavior
        real_action += np.random.normal(0, self.calibration_params['pressure_noise_std'], real_action.shape)
        
        return real_action 