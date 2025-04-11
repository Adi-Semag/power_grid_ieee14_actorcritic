import numpy as np
import gym
from gym import spaces
import math

class PowerGridEnv(gym.Env):
    def __init__(self):
        super(PowerGridEnv, self).__init__()
        
        # IEEE 14-bus system parameters
        self.n_buses = 14
        self.n_generators = 14  # All buses can now act as generators
        self.generator_buses = list(range(1, 15))  # All buses are potential generators
        
        # Define action space with improved bounds
        # [V1-V14, P1-P14, Q1-Q14]
        self.action_space = spaces.Box(
            low=np.array([-0.1] * 14 + [-0.15] * 14 + [-0.15] * 14),  # Tighter bounds for better control
            high=np.array([0.1] * 14 + [0.15] * 14 + [0.15] * 14),
            dtype=np.float32
        )
        
        # Define state space with improved bounds
        # [V1-V14, θ1-θ14, P1-P14, Q1-Q14]
        self.state_space = spaces.Box(
            low=np.array([0.8] * 14 + [-np.pi/2] * 14 + [-1.5] * 14 + [-1.5] * 14),  # Tighter bounds
            high=np.array([1.2] * 14 + [np.pi/2] * 14 + [1.5] * 14 + [1.5] * 14),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.step_counter = 0
        self.max_steps = 300
        
        # Curriculum learning parameters with improved stages
        self.curriculum_stage = 0
        self.max_curriculum_stages = 5
        self.curriculum_thresholds = {
            0: {'voltage_weight': 3.0, 'angle_weight': 1.0, 'power_loss_weight': 0.1, 'reward_threshold': -400},
            1: {'voltage_weight': 4.0, 'angle_weight': 1.5, 'power_loss_weight': 0.2, 'reward_threshold': -300},
            2: {'voltage_weight': 5.0, 'angle_weight': 2.0, 'power_loss_weight': 0.3, 'reward_threshold': -200},
            3: {'voltage_weight': 6.0, 'angle_weight': 2.5, 'power_loss_weight': 0.4, 'reward_threshold': -150},
            4: {'voltage_weight': 8.0, 'angle_weight': 3.0, 'power_loss_weight': 0.5, 'reward_threshold': -100}
        }
        self.curriculum_progress = {
            'episodes_at_stage': 0,
            'best_reward_at_stage': float('-inf'),
            'stable_episodes': 0
        }
        
        # Smooth transition parameters
        self.transition_progress = 0.0  # 0.0 to 1.0
        self.transition_rate = 0.05  # How quickly to transition between stages
        
        # Load data (simplified for this example)
        self.load_data = {
            1: {'P': 0.0, 'Q': 0.0},  # Slack bus
            2: {'P': 0.217, 'Q': 0.127},
            3: {'P': 0.942, 'Q': 0.19},
            4: {'P': 0.478, 'Q': -0.039},
            5: {'P': 0.076, 'Q': -0.016},
            6: {'P': 0.112, 'Q': 0.075},
            7: {'P': 0.0, 'Q': 0.0},
            8: {'P': 0.0, 'Q': 0.0},
            9: {'P': 0.295, 'Q': -0.046},
            10: {'P': 0.09, 'Q': 0.058},
            11: {'P': 0.035, 'Q': 0.018},
            12: {'P': 0.061, 'Q': 0.016},
            13: {'P': 0.135, 'Q': 0.058},
            14: {'P': 0.149, 'Q': 0.05}
        }
        
        # Generator limits with improved differentiation
        self.gen_limits = {}
        for bus in range(1, 15):
            if bus in [1, 2, 3, 6, 8]:  # Original generator buses
                self.gen_limits[bus] = {
                    'Pmin': -1.0, 'Pmax': 1.0,
                    'Qmin': -1.0, 'Qmax': 1.0,
                    'Vmin': 0.95, 'Vmax': 1.05,
                    'ramp_rate': 0.1  # Added ramp rate limit
                }
            else:  # New potential generator buses
                self.gen_limits[bus] = {
                    'Pmin': -0.2, 'Pmax': 0.2,
                    'Qmin': -0.2, 'Qmax': 0.2,
                    'Vmin': 0.95, 'Vmax': 1.05,
                    'ramp_rate': 0.05  # Lower ramp rate for new generators
                }
        
        # Branch data (simplified)
        self.branch_data = [
            (1, 2), (1, 5), (1, 8), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5),
            (4, 7), (4, 9), (5, 6), (6, 11), (6, 12), (6, 13), (7, 8), (7, 9),
            (9, 10), (9, 14), (10, 11), (12, 13), (13, 14)
        ]
        
        # Branch parameters (simplified)
        self.branch_params = {
            (1, 2): {'r': 0.05917, 'x': 0.22304},
            (1, 5): {'r': 0.22304, 'x': 0.08462},
            (1, 8): {'r': 0.08462, 'x': 0.22304},
            (2, 3): {'r': 0.19797, 'x': 0.17632},
            (2, 4): {'r': 0.17632, 'x': 0.17388},
            (2, 5): {'r': 0.17388, 'x': 0.17103},
            (3, 4): {'r': 0.17103, 'x': 0.19207},
            (4, 5): {'r': 0.19207, 'x': 0.19999},
            (4, 7): {'r': 0.19999, 'x': 0.17615},
            (4, 9): {'r': 0.17615, 'x': 0.17388},
            (5, 6): {'r': 0.17388, 'x': 0.17103},
            (6, 11): {'r': 0.17103, 'x': 0.19207},
            (6, 12): {'r': 0.19207, 'x': 0.19999},
            (6, 13): {'r': 0.19999, 'x': 0.17615},
            (7, 8): {'r': 0.17615, 'x': 0.17388},
            (7, 9): {'r': 0.17388, 'x': 0.17103},
            (9, 10): {'r': 0.17103, 'x': 0.19207},
            (9, 14): {'r': 0.19207, 'x': 0.19999},
            (10, 11): {'r': 0.19999, 'x': 0.17615},
            (12, 13): {'r': 0.17615, 'x': 0.17388},
            (13, 14): {'r': 0.17388, 'x': 0.17103}
        }
        
        # Initialize observation space
        self.observation_space = self.state_space
        
        # Reward weights with improved balance
        self.voltage_weight = 8.0
        self.angle_weight = 3.0
        self.power_loss_weight = 0.5
        self.action_weight = 0.2
        self.power_balance_weight = 3.0
        self.stability_weight = 5.0
        
        # Add state normalization parameters
        self.state_mean = np.zeros(4 * self.n_buses)
        self.state_std = np.ones(4 * self.n_buses)
        self.state_mean[:self.n_buses] = 1.0  # Nominal voltage
        self.state_std[:self.n_buses] = 0.1   # Voltage std
        self.state_std[self.n_buses:2*self.n_buses] = np.pi/4  # Angle std
        self.state_std[2*self.n_buses:] = 0.5  # Power std
        
        # Hierarchical control parameters
        self.voltage_control_mode = True  # Start with voltage control
        self.voltage_control_episodes = 200  # Episodes to focus on voltage before switching
        
    def reset(self):
        # Initialize state with nominal values
        voltages = np.ones(self.n_buses)
        angles = np.zeros(self.n_buses)
        
        # Initialize generator outputs
        gen_P = np.zeros(self.n_generators)
        gen_Q = np.zeros(self.n_generators)
        
        # Set initial generator values
        for i, bus in enumerate(self.generator_buses):
            gen_P[i] = 0.5  # Start at half capacity
            gen_Q[i] = 0.0
        
        # Combine into state
        self.state = np.concatenate([voltages, angles, gen_P, gen_Q])
        
        # Reset step counter
        self.step_counter = 0
        
        # Reset curriculum stage for new training run
        self.curriculum_stage = 0
        self.transition_progress = 0.0
        self.curriculum_progress = {
            'episodes_at_stage': 0,
            'best_reward_at_stage': float('-inf'),
            'stable_episodes': 0
        }
        
        # Reset hierarchical control
        self.voltage_control_mode = True
        
        return self.state
    
    def step(self, action):
        self.step_counter += 1
        
        # Apply action with curriculum-based scaling
        scaled_action = self._scale_action(action)
        
        # Update state based on action
        new_state = self._update_state(scaled_action)
        
        # Calculate power flow
        voltages, angles, power_loss = self._power_flow(new_state)
        
        # Update state with power flow results
        new_state[:self.n_buses] = voltages
        new_state[self.n_buses:2*self.n_buses] = angles
        
        # Calculate reward
        reward = self._calculate_reward(new_state, power_loss, scaled_action)
        
        # Check if episode is done
        done = self._check_termination(new_state, power_loss)
        
        # Update state
        self.state = new_state
        
        # Update curriculum learning
        self._update_curriculum_stage(reward)
        
        # Update hierarchical control mode
        if self.voltage_control_mode and self.step_counter > self.voltage_control_episodes:
            self.voltage_control_mode = False
        
        return self.state, reward, done, {}
    
    def _scale_action(self, action):
        # Get current curriculum parameters
        action_range, voltage_tolerance = self._get_curriculum_parameters()
        
        # Apply smooth transition between stages
        if self.transition_progress < 1.0:
            prev_range, prev_tolerance = self._get_curriculum_parameters(self.curriculum_stage - 1)
            action_range = prev_range + (action_range - prev_range) * self.transition_progress
            voltage_tolerance = prev_tolerance + (voltage_tolerance - prev_tolerance) * self.transition_progress
        
        # Scale action based on curriculum stage
        scaled_action = action * action_range
        
        # Apply hierarchical control if in voltage control mode
        if self.voltage_control_mode:
            # Focus actions on voltage control (first 14 components)
            scaled_action[14:] *= 0.1  # Reduce impact of power control actions
        
        return scaled_action
    
    def _get_curriculum_parameters(self, stage=None):
        if stage is None:
            stage = self.curriculum_stage
        
        # Enhanced curriculum stages with more granular progression
        if stage == 0:
            return 0.05, 0.1  # Basic voltage control
        elif stage == 1:
            return 0.08, 0.08  # Intermediate voltage control
        elif stage == 2:
            return 0.12, 0.06  # Advanced voltage control
        elif stage == 3:
            return 0.15, 0.04  # Basic power flow
        elif stage == 4:
            return 0.18, 0.03  # Intermediate power flow
        else:  # stage 5
            return 0.2, 0.02   # Full complexity
        
    def _update_state(self, action):
        new_state = self.state.copy()
        
        # Update generator values based on action
        # First 14 actions: voltage setpoints
        # Next 14 actions: active power
        # Last 14 actions: reactive power
        for i, bus in enumerate(self.generator_buses):
            # Update voltage setpoint
            new_state[i] += action[i]
            
            # Update active power
            new_state[self.n_buses + i] += action[i+14]
            
            # Update reactive power
            new_state[self.n_buses + self.n_generators + i] += action[i+28]
        
        # Ensure values stay within bounds
        new_state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
        
        return new_state
    
    def _power_flow(self, state):
        # Extract state components
        voltages = state[:self.n_buses]
        angles = state[self.n_buses:2*self.n_buses]
        gen_P = state[2*self.n_buses:2*self.n_buses+self.n_generators]
        gen_Q = state[2*self.n_buses+self.n_generators:]
        
        # Improved power flow calculation with adaptive convergence
        max_iter = 50  # Increased from 30
        base_tolerance = 1e-10
        min_tolerance = 1e-12
        max_tolerance = 1e-8
        
        # Initialize power mismatch with dynamic scaling
        P_mismatch = np.zeros(self.n_buses)
        Q_mismatch = np.zeros(self.n_buses)
        
        # Calculate initial power mismatches with load scaling
        total_load = sum(self.load_data[bus]['P'] for bus in range(1, self.n_buses+1))
        load_scale = 1.0 / (total_load + 1e-10)
        
        for i in range(self.n_buses):
            bus = i + 1
            if bus in self.generator_buses:
                gen_idx = self.generator_buses.index(bus)
                P_mismatch[i] = (gen_P[gen_idx] - self.load_data[bus]['P']) * load_scale
                Q_mismatch[i] = (gen_Q[gen_idx] - self.load_data[bus]['Q']) * load_scale
            else:
                P_mismatch[i] = -self.load_data[bus]['P'] * load_scale
                Q_mismatch[i] = -self.load_data[bus]['Q'] * load_scale
        
        # Iterative power flow solution with adaptive tolerance
        for iter in range(max_iter):
            # Calculate power flows with improved branch handling
            P_flow = np.zeros(self.n_buses)
            Q_flow = np.zeros(self.n_buses)
            
            for from_bus, to_bus in self.branch_data:
                from_idx = from_bus - 1
                to_idx = to_bus - 1
                
                # Get branch parameters with improved numerical stability
                r = self.branch_params[(from_bus, to_bus)]['r']
                x = self.branch_params[(from_bus, to_bus)]['x']
                z_squared = max(r*r + x*x, 1e-10)  # Prevent division by zero
                y = 1 / z_squared
                g = r * y
                b = -x * y
                
                # Calculate power flow with improved numerical stability
                V_from = max(voltages[from_idx], 0.1)  # Prevent zero voltage
                V_to = max(voltages[to_idx], 0.1)
                theta_from = angles[from_idx]
                theta_to = angles[to_idx]
                angle_diff = theta_from - theta_to
                
                # Active power flow with improved stability
                P_from = V_from**2 * g - V_from * V_to * (g * np.cos(angle_diff) + b * np.sin(angle_diff))
                P_to = V_to**2 * g - V_from * V_to * (g * np.cos(-angle_diff) + b * np.sin(-angle_diff))
                
                # Reactive power flow with improved stability
                Q_from = -V_from**2 * (b + 0.5 * b) + V_from * V_to * (b * np.cos(angle_diff) - g * np.sin(angle_diff))
                Q_to = -V_to**2 * (b + 0.5 * b) + V_from * V_to * (b * np.cos(-angle_diff) - g * np.sin(-angle_diff))
                
                # Update power flows with smoothing
                P_flow[from_idx] += P_from
                P_flow[to_idx] += P_to
                Q_flow[from_idx] += Q_from
                Q_flow[to_idx] += Q_to
            
            # Calculate power mismatches with adaptive tolerance
            P_mismatch_new = P_mismatch - P_flow
            Q_mismatch_new = Q_mismatch - Q_flow
            
            # Adaptive tolerance based on iteration progress
            current_tolerance = max(min_tolerance, 
                                 min(max_tolerance, 
                                     base_tolerance * (1 + iter/max_iter)))
            
            # Check convergence with improved criteria
            P_converged = np.max(np.abs(P_mismatch_new)) < current_tolerance
            Q_converged = np.max(np.abs(Q_mismatch_new)) < current_tolerance
            
            if P_converged and Q_converged:
                break
            
            # Update voltages and angles with adaptive step size
            for i in range(self.n_buses):
                # Adaptive step size based on mismatch magnitude and iteration
                base_step = 0.1 * (1 - iter/max_iter)  # Decreasing step size
                P_step = min(base_step, 0.01 / (np.abs(P_mismatch_new[i]) + 1e-10))
                Q_step = min(base_step, 0.01 / (np.abs(Q_mismatch_new[i]) + 1e-10))
                
                # Update angles (for active power)
                if i > 0:  # Skip slack bus
                    angles[i] += P_step * P_mismatch_new[i] / max(voltages[i], 0.1)
                
                # Update voltages (for reactive power)
                if i not in [0, 3, 6, 8]:  # Skip generator buses
                    voltages[i] += Q_step * Q_mismatch_new[i] / max(voltages[i], 0.1)
            
            # Update mismatches with smoothing
            P_mismatch = 0.7 * P_mismatch + 0.3 * P_mismatch_new
            Q_mismatch = 0.7 * Q_mismatch + 0.3 * Q_mismatch_new
        
        # Calculate total power loss with improved accuracy
        power_loss = np.sum(np.abs(P_flow)) + np.sum(np.abs(Q_flow))
        
        return voltages, angles, power_loss
    
    def _calculate_reward(self, state, power_loss, action):
        # Extract state components
        voltages = state[:self.n_buses]
        angles = state[self.n_buses:2*self.n_buses]
        
        # Get current curriculum parameters
        _, voltage_tolerance = self._get_curriculum_parameters()
        
        # Calculate voltage deviation from nominal (1.0 p.u.)
        voltage_deviation = np.mean(np.abs(voltages - 1.0))
        
        # Calculate angle deviation from nominal (0.0 radians)
        angle_deviation = np.mean(np.abs(angles))
        
        # Calculate action magnitude penalty
        action_penalty = np.mean(np.abs(action))
        
        # Calculate stability reward (penalize large angle differences)
        max_angle_diff = 0
        for from_bus, to_bus in self.branch_data:
            from_idx = from_bus - 1
            to_idx = to_bus - 1
            angle_diff = abs(angles[from_idx] - angles[to_idx])
            max_angle_diff = max(max_angle_diff, angle_diff)
        
        # Calculate power balance reward
        gen_P = state[2*self.n_buses:2*self.n_buses+self.n_generators]
        total_gen_P = np.sum(gen_P)
        total_load_P = sum(self.load_data[bus]['P'] for bus in range(1, self.n_buses+1))
        power_balance = abs(total_gen_P - total_load_P)
        
        # Calculate reward components with curriculum-based weights
        voltage_reward = -self.voltage_weight * voltage_deviation / voltage_tolerance
        angle_reward = -self.angle_weight * angle_deviation
        power_loss_reward = -self.power_loss_weight * power_loss
        action_reward = -self.action_weight * action_penalty
        stability_reward = -self.stability_weight * max_angle_diff
        power_balance_reward = -self.power_balance_weight * power_balance
        
        # Add small positive reward for maintaining voltage within limits
        if voltage_deviation < voltage_tolerance:
            voltage_reward += 0.5
        
        # Combine rewards
        reward = voltage_reward + angle_reward + power_loss_reward + action_reward + stability_reward + power_balance_reward
        
        # Apply hierarchical control reward adjustment
        if self.voltage_control_mode:
            # Emphasize voltage control during initial episodes
            reward = 2.0 * voltage_reward + 0.5 * (angle_reward + power_loss_reward + action_reward + stability_reward + power_balance_reward)
        
        return reward
    
    def _check_termination(self, state, power_loss):
        # Check if maximum steps reached
        if self.step_counter >= self.max_steps:
            return True
        
        # Extract state components
        voltages = state[:self.n_buses]
        angles = state[self.n_buses:2*self.n_buses]
        gen_P = state[2*self.n_buses:2*self.n_buses+self.n_generators]
        gen_Q = state[2*self.n_buses+self.n_generators:]
        
        # Get current curriculum parameters
        _, voltage_tolerance = self._get_curriculum_parameters()
        
        # Check for voltage violations
        for i, voltage in enumerate(voltages):
            if abs(voltage - 1.0) > voltage_tolerance:
                return True
        
        # Check for angle violations
        for angle in angles:
            if abs(angle) > np.pi/2:  # 90 degrees
                return True
        
        # Check for generator limit violations
        for i, bus in enumerate(self.generator_buses):
            if (gen_P[i] < self.gen_limits[bus]['Pmin'] or 
                gen_P[i] > self.gen_limits[bus]['Pmax'] or
                gen_Q[i] < self.gen_limits[bus]['Qmin'] or 
                gen_Q[i] > self.gen_limits[bus]['Qmax']):
                return True
        
        return False
    
    def _update_curriculum_stage(self, reward):
        """Update curriculum stage based on performance"""
        current_threshold = self.curriculum_thresholds[self.curriculum_stage]['reward_threshold']
        
        # Update progress tracking
        self.curriculum_progress['episodes_at_stage'] += 1
        
        # Update best reward for current stage
        if reward > self.curriculum_progress['best_reward_at_stage']:
            self.curriculum_progress['best_reward_at_stage'] = reward
            self.curriculum_progress['stable_episodes'] = 0
        else:
            self.curriculum_progress['stable_episodes'] += 1
        
        # Check if ready to advance
        if (self.curriculum_progress['best_reward_at_stage'] > current_threshold and 
            self.curriculum_progress['stable_episodes'] >= 20 and  # Must maintain performance
            self.curriculum_progress['episodes_at_stage'] >= 100):  # Minimum episodes per stage
            
            if self.curriculum_stage < self.max_curriculum_stages - 1:
                self.curriculum_stage += 1
                self.curriculum_progress = {
                    'episodes_at_stage': 0,
                    'best_reward_at_stage': float('-inf'),
                    'stable_episodes': 0
                }
                return True
        
        return False

    def calculate_power_flow(self, state):
        """Calculate power flow using simplified DC power flow equations"""
        voltages = state[:self.n_buses]
        angles = state[self.n_buses:2*self.n_buses]
        P_gen = state[2*self.n_buses:2*self.n_buses+self.n_generators]
        Q_gen = state[2*self.n_buses+self.n_generators:]
        
        # Initialize power flow results
        P_flow = np.zeros((self.n_buses, self.n_buses))
        Q_flow = np.zeros((self.n_buses, self.n_buses))
        
        # Calculate power flows for each branch
        for from_bus, to_bus in self.branch_data:
            if (from_bus, to_bus) in self.branch_params:
                params = self.branch_params[(from_bus, to_bus)]
                r, x = params['r'], params['x']
                z_squared = r*r + x*x
                
                # DC power flow approximation
                angle_diff = angles[from_bus-1] - angles[to_bus-1]
                P_flow[from_bus-1, to_bus-1] = (1/z_squared) * (r * np.cos(angle_diff) + x * np.sin(angle_diff))
                P_flow[to_bus-1, from_bus-1] = -P_flow[from_bus-1, to_bus-1]
                
                # AC power flow components
                V_from = voltages[from_bus-1]
                V_to = voltages[to_bus-1]
                
                # Active power flow
                P_ac = (V_from * V_from * r - V_from * V_to * (r * np.cos(angle_diff) + x * np.sin(angle_diff))) / z_squared
                P_flow[from_bus-1, to_bus-1] = P_ac
                P_flow[to_bus-1, from_bus-1] = -P_ac
                
                # Reactive power flow
                Q_ac = (V_from * V_from * x - V_from * V_to * (x * np.cos(angle_diff) - r * np.sin(angle_diff))) / z_squared
                Q_flow[from_bus-1, to_bus-1] = Q_ac
                Q_flow[to_bus-1, from_bus-1] = -Q_ac
        
        # Calculate power balance
        P_balance = np.zeros(self.n_buses)
        Q_balance = np.zeros(self.n_buses)
        
        for bus in range(1, self.n_buses + 1):
            # Active power balance
            P_injection = P_gen[bus-1] if bus in self.generator_buses else 0
            P_load = self.load_data[bus]['P']
            P_balance[bus-1] = P_injection - P_load - np.sum(P_flow[bus-1, :])
            
            # Reactive power balance
            Q_injection = Q_gen[bus-1] if bus in self.generator_buses else 0
            Q_load = self.load_data[bus]['Q']
            Q_balance[bus-1] = Q_injection - Q_load - np.sum(Q_flow[bus-1, :])
        
        return P_flow, Q_flow, P_balance, Q_balance

    def calculate_reward(self, state, action):
        """Calculate reward with improved shaping and stability metrics"""
        # Get power flow results
        P_flow, Q_flow, P_balance, Q_balance = self.calculate_power_flow(state)
        
        # Extract state components
        voltages = state[:self.n_buses]
        angles = state[self.n_buses:2*self.n_buses]
        P_gen = state[2*self.n_buses:2*self.n_buses+self.n_generators]
        Q_gen = state[2*self.n_buses+self.n_generators:]
        
        # Progressive reward weights based on performance
        base_weights = {
            'voltage': self.voltage_weight,
            'angle': self.angle_weight,
            'power_loss': self.power_loss_weight,
            'power_balance': self.power_balance_weight,
            'action': self.action_weight,
            'stability': self.stability_weight
        }
        
        # Adaptive weights based on constraint violations
        voltage_violations = np.sum(np.abs(voltages - 1.0) > 0.05)
        angle_violations = np.sum(np.abs(angles) > np.pi/4)
        
        weights = base_weights.copy()
        if voltage_violations > 0:
            weights['voltage'] *= (1 + 0.2 * voltage_violations)
        if angle_violations > 0:
            weights['angle'] *= (1 + 0.2 * angle_violations)
        
        # Voltage stability reward with progressive thresholds
        voltage_deviation = np.mean(np.abs(voltages - 1.0))
        voltage_reward = -weights['voltage'] * voltage_deviation
        if voltage_deviation < 0.03:  # Excellent control
            voltage_reward += 2.0
        elif voltage_deviation < 0.05:  # Good control
            voltage_reward += 1.0
        
        # Angle stability with topology consideration
        angle_deviation = np.mean(np.abs(angles))
        max_angle_diff = max(abs(angles[i] - angles[j]) 
                           for i, j in self.branch_data)
        angle_reward = -weights['angle'] * (angle_deviation + 0.5 * max_angle_diff)
        
        # Power flow optimization with loss minimization
        power_loss = np.sum(np.abs(P_flow)) + np.sum(np.abs(Q_flow))
        normalized_loss = power_loss / (self.n_buses * 2)  # Normalize by system size
        power_loss_reward = -weights['power_loss'] * normalized_loss
        
        # Power balance with progressive penalties
        power_imbalance = np.mean(np.abs(P_balance)) + np.mean(np.abs(Q_balance))
        power_balance_reward = -weights['power_balance'] * power_imbalance
        if power_imbalance < 0.01:  # Excellent balance
            power_balance_reward += 2.0
        elif power_imbalance < 0.05:  # Good balance
            power_balance_reward += 1.0
        
        # Generator utilization and efficiency
        gen_utilization = np.array([abs(P_gen[i]) / self.gen_limits[bus]['Pmax'] 
                                  for i, bus in enumerate(self.generator_buses)])
        efficiency_reward = 0.5 * (1 - np.std(gen_utilization))  # Reward balanced usage
        
        # Ramp rate compliance
        prev_P_gen = self.state[2*self.n_buses:2*self.n_buses+self.n_generators]
        ramp_violations = sum(1 for i, bus in enumerate(self.generator_buses)
                            if abs(P_gen[i] - prev_P_gen[i]) > self.gen_limits[bus]['ramp_rate'])
        ramp_penalty = -2.0 * ramp_violations
        
        # Stability reward with comprehensive metrics
        stability_reward = 0
        if (voltage_deviation < 0.05 and 
            angle_deviation < 0.1 and 
            power_imbalance < 0.05 and
            max_angle_diff < np.pi/6 and  # 30 degrees
            ramp_violations == 0):
            stability_reward = weights['stability'] * (1 - voltage_deviation)
        
        # Action smoothness reward
        action_smoothness = -weights['action'] * (
            np.mean(np.square(action)) + 
            0.5 * np.mean(np.abs(np.diff(action)))  # Penalize rapid changes
        )
        
        # Total reward with progressive shaping
        total_reward = (voltage_reward + angle_reward + power_loss_reward + 
                       power_balance_reward + action_smoothness + stability_reward +
                       efficiency_reward + ramp_penalty)
        
        # Curriculum learning with smoother transitions
        if self.curriculum_stage < self.max_curriculum_stages:
            stage_params = self.curriculum_thresholds[self.curriculum_stage]
            # Smooth transition factor
            progress = min(1.0, self.curriculum_progress['episodes_at_stage'] / stage_params['min_episodes'])
            transition_factor = 1 + progress * (
                stage_params['voltage_weight'] / self.voltage_weight - 1
            )
            total_reward *= transition_factor
            
            # Stage-specific emphasis
            if self.curriculum_stage == 0:  # Focus on voltage control
                total_reward = 2.0 * voltage_reward + 0.5 * (total_reward - voltage_reward)
            elif self.curriculum_stage == 1:  # Focus on power balance
                total_reward = 1.5 * power_balance_reward + 0.7 * (total_reward - power_balance_reward)
            elif self.curriculum_stage == 2:  # Focus on stability
                total_reward = 1.5 * stability_reward + 0.7 * (total_reward - stability_reward)
        
        return total_reward 