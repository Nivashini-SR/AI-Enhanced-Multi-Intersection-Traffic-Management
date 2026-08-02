import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

SUMO_CONFIG = "simulation.sumocfg"

class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.step_count = 0
        self.max_steps = 600
        self.min_green = 10
        self.last_switch = {}
        self.max_tls = 10

        self.action_space = spaces.MultiDiscrete([2] * self.max_tls)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.max_tls * 6,), dtype=np.float32
        )

        self.tls_list = []
        self.num_tls = 0
        
        # New Cache Systems
        self.active_ambulances = set()
        self.tls_lanes = []
        self.tls_lane_map = {}
        self.tls_phase_map = {}
        self.priority_locks = {}
        self.tls_current_target = {}
        self.tls_yellow_active = {}
        self.tls_yellow_timer = {}
        self.tls_yellow_phase = {}

        # --delay slows the GUI during demos (value in ms); has no effect on headless training
        DEMO_DELAY_MS = 200  # ← increase this (e.g. 500) to slow the GUI further
        if self.render_mode == "human":
            self.sumo_cmd = ["sumo-gui", "-c", SUMO_CONFIG, "--no-step-log", "--no-warnings",
                             "--ignore-route-errors", "true", "--delay", str(DEMO_DELAY_MS)]
        else:
            self.sumo_cmd = ["sumo", "-c", SUMO_CONFIG, "--no-step-log", "--no-warnings",
                             "--ignore-route-errors", "true"]
        
        
        import traci as traci_module
        
        self.traci = traci_module

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            self.traci.close()
        except Exception:
            pass

        self.traci.start(self.sumo_cmd)
        
        if self.render_mode == "human":
            os.makedirs("screenshots", exist_ok=True)

        self.step_count = 0
        self.tls_list = self.traci.trafficlight.getIDList()
        self.num_tls = len(self.tls_list)

        self.last_switch = {tls: 0 for tls in self.tls_list}
        
        if self.render_mode == "human":
            try:
                # Immediate Visibility: Focus on the major central junction
                # Coordinates from research: (1324, 892)
                self.traci.gui.setOffset("View #0", 1324, 892)
                self.traci.gui.setZoom("View #0", 800)
            except Exception:
                pass
        
        # Build strict lane caches for O(1) lookups
        self.tls_lanes = []
        self.tls_lane_map = {}
        self.tls_phase_map = {}
        self.direction_lane_map = {} # Maps tls -> {direction_id: [lanes]}
        
        # Filter to only keep the first 10 junctions for active AI control
        self.active_tls_list = self.tls_list[:self.max_tls]
        
        for tls in self.active_tls_list:
            try:
                lanes = list(set(self.traci.trafficlight.getControlledLanes(tls)))
                self.tls_lane_map[tls] = lanes
                self.tls_lanes.extend(lanes)

                # Analyze TL Logic to map actions 0 and 1 to major Green phases
                logic = self.traci.trafficlight.getAllProgramLogics(tls)[0]
                phases = logic.phases
                # Find the two most distinct "Greenest" phases
                green_phases = []
                for idx, p in enumerate(phases):
                    # Count 'G' or 'g' in state
                    score = p.state.lower().count('g')
                    if score > 0:
                        green_phases.append((idx, p.state, score))
                
                if len(green_phases) >= 2:
                    # Store the first one as action 0
                    p0_idx, p0_state, _ = green_phases[0]
                    # Find the most different one as action 1
                    best_diff = -1
                    p1_idx, p1_state = green_phases[-1][0], green_phases[-1][1]
                    for idx, state, score in green_phases[1:]:
                        diff = sum(1 for a, b in zip(p0_state, state) if a != b)
                        if diff > best_diff:
                            best_diff = diff
                            p1_idx = idx
                            p1_state = state
                    self.tls_phase_map[tls] = {0: p0_idx, 1: p1_idx}
                    
                    # Directional Lane Mapping: Identify which lanes belong to Action 0 (N-S) vs Action 1 (E-W)
                    # Controlled links gives us mapping of link indices to lanes
                    links = self.traci.trafficlight.getControlledLinks(tls)
                    self.direction_lane_map[tls] = {0: set(), 1: set()}
                    for dir_id, p_state in [(0, p0_state), (1, p1_state)]:
                        for idx, link_data in enumerate(links):
                            if idx < len(p_state) and p_state[idx].lower() == 'g':
                                for link in link_data:
                                    # link[0] is the incoming lane
                                    self.direction_lane_map[tls][dir_id].add(link[0])
                    
                    # Convert sets to lists
                    self.direction_lane_map[tls][0] = list(self.direction_lane_map[tls][0])
                    self.direction_lane_map[tls][1] = list(self.direction_lane_map[tls][1])

                    # Find Yellow Phases between Green 0 and Green 1
                    self.tls_yellow_phase[tls] = {}
                    for g_from in [0, 1]:
                        g_to = 1 - g_from
                        p_from = self.tls_phase_map[tls][g_from]
                        # Find a yellow phase 'y' after p_from
                        y_idx = (p_from + 1) % len(phases)
                        if 'y' in phases[y_idx].state.lower():
                            self.tls_yellow_phase[tls][g_from] = y_idx
                        else:
                            for idx, p in enumerate(phases):
                                if 'y' in p.state.lower():
                                    self.tls_yellow_phase[tls][g_from] = idx
                                    break
                    
                    self.tls_current_target[tls] = 0
                    self.tls_yellow_active[tls] = False
                    self.tls_yellow_timer[tls] = 0
                else:
                    self.tls_phase_map[tls] = {0: 0, 1: 0} 
                    self.tls_yellow_phase[tls] = {0: 0, 1: 0}
                    self.direction_lane_map[tls] = {0: [], 1: []}
                    self.tls_current_target[tls] = 0
                    self.tls_yellow_active[tls] = False
                    self.tls_yellow_timer[tls] = 0
            except Exception as e:
                print(f"Skipping flaky junction {tls}: {e}")
                continue
        
        # Deduplicate globally controlled lanes
        self.tls_lanes = list(set(self.tls_lanes))
        
        # Reset vehicles
        self.active_ambulances = set()
        self.priority_locks = {}

        # Universal monitoring of ALL TLS in the map
        self.tls_list = self.traci.trafficlight.getIDList()
        self.num_tls = len(self.tls_list)

        return self.get_state(), {}

    def get_state(self):
        state = []
        
        # Pre-calculate nearest ambulance per junction for O(N)
        amb_positions = []
        for amb in self.active_ambulances:
            try:
                amb_positions.append(self.traci.vehicle.getPosition(amb))
            except Exception:
                pass

        for tls in self.active_tls_list:
            try:
                # 1 & 2: Waiting Vehicles per Direction
                lanes_0 = self.direction_lane_map.get(tls, {}).get(0, [])
                lanes_1 = self.direction_lane_map.get(tls, {}).get(1, [])
                
                wait_0 = sum(self.traci.lane.getLastStepHaltingNumber(l) for l in lanes_0 if "crossing" not in l)
                wait_1 = sum(self.traci.lane.getLastStepHaltingNumber(l) for l in lanes_1 if "crossing" not in l)
                
                # 3 & 4: Waiting Pedestrians per Direction
                ped_0 = sum(self.traci.lane.getLastStepPersonNumber(l) for l in lanes_0 if "crossing" in l or "sidewalk" in l)
                ped_1 = sum(self.traci.lane.getLastStepPersonNumber(l) for l in lanes_1 if "crossing" in l or "sidewalk" in l)
                
                # 5: Current phase
                phase = self.traci.trafficlight.getPhase(tls)
                
                # 6: Ambulance Radar (Distance to nearest ambulance, 1.0 if none, 0.0 if right at junction)
                amb_dist = 1.0
                if amb_positions:
                    jx, jy = self.traci.junction.getPosition(tls)
                    min_d = min(np.sqrt((jx-ax)**2 + (jy-ay)**2) for ax, ay in amb_positions)
                    amb_dist = min(1.0, min_d / 300.0) # Normalized to 300m
                
                state.extend([wait_0, wait_1, ped_0, ped_1, phase, amb_dist])
            except Exception:
                state.extend([0, 0, 0, 0, 0, 1.0])

        while len(state) < self.max_tls * 6:
            state.extend([0, 0, 0, 0, 0, 1.0])

        return np.array(state[:self.max_tls * 6], dtype=np.float32) / 100.0

    def step(self, action):
        
        # Update cache O(K) instead of O(N)
        newly_spawned = self.traci.simulation.getDepartedIDList()
        for v in newly_spawned:
            try:
                if self.traci.vehicle.getTypeID(v) == "ambulance":
                    self.active_ambulances.add(v)
            except Exception:
                pass
                
        newly_arrived = self.traci.simulation.getArrivedIDList()
        for v in newly_arrived:
            self.active_ambulances.discard(v)

        for v in list(self.active_ambulances):
            try:
                if self.render_mode == "human":
                    try:
                        self.traci.gui.trackVehicle("View #0", v)
                    except Exception:
                        pass

                next_tls = self.traci.vehicle.getNextTLS(v)
                if next_tls:
                    tls_id, index, distance, _ = next_tls[0]
                    if distance < 250:
                        logic = self.traci.trafficlight.getAllProgramLogics(tls_id)[0]
                        for p_idx, p in enumerate(logic.phases):
                            if index < len(p.state) and p.state[index].lower() == 'g':
                                self.traci.trafficlight.setPhase(tls_id, p_idx)
                                self.priority_locks[tls_id] = self.step_count + 10 
                                break

                # New: Active Route Clearing Logic
                lane_id = self.traci.vehicle.getLaneID(v)
                pos = self.traci.vehicle.getLanePosition(v)
                # Find other vehicles on the same lane ahead of the ambulance
                others = self.traci.lane.getLastStepVehicleIDs(lane_id)
                for other_v in others:
                    if other_v == v: continue
                    other_pos = self.traci.vehicle.getLanePosition(other_v)
                    # If vehicle is within 50 meters ahead
                    if 0 < (other_pos - pos) < 50:
                        try:
                            # 1. Try to push it to a different lane if possible
                            num_lanes = self.traci.edge.getLaneNumber(self.traci.vehicle.getRoadID(v))
                            if num_lanes > 1:
                                current_lane = self.traci.vehicle.getLaneIndex(other_v)
                                target_lane = (current_lane + 1) % num_lanes
                                self.traci.vehicle.changeLane(other_v, target_lane, 2.0)
                            
                            # 2. Even if it can't change lane, nudge its speed factor to move it faster out of the way
                            self.traci.vehicle.setSpeedFactor(other_v, 1.3)
                            self.traci.vehicle.setSpeedMode(other_v, 0) # Disable safety checks temporarily to clear bottleneck
                        except Exception:
                            pass
            except Exception:
                self.active_ambulances.discard(v)
                continue

        for i, tls in enumerate(self.active_tls_list):
            try:
                if i >= len(action):
                    break
                
                # Skip if intersection is currently locked for an ambulance
                if self.priority_locks.get(tls, 0) > self.step_count:
                    continue

                target_action = int(action[i])
                
                # 1. Handle Active Yellow Logic
                if self.tls_yellow_active.get(tls, False):
                    if self.tls_yellow_timer[tls] > 0:
                        self.tls_yellow_timer[tls] -= 1
                        continue
                    else:
                        # Transition to final green
                        self.tls_yellow_active[tls] = False
                        self.traci.trafficlight.setPhase(tls, self.tls_phase_map[tls][target_action])
                        self.tls_current_target[tls] = target_action
                        continue

                # 2. Check if Model wants to switch
                if target_action != self.tls_current_target.get(tls, 0):
                    if self.step_count - self.last_switch.get(tls, 0) > self.min_green:
                        try:
                            # Start yellow transition
                            y_phase = self.tls_yellow_phase[tls].get(self.tls_current_target[tls], None)
                            if y_phase is not None:
                                self.traci.trafficlight.setPhase(tls, y_phase)
                                self.tls_yellow_active[tls] = True
                                self.tls_yellow_timer[tls] = 3 # 3-step yellow buffer
                            else:
                                # Direct green switch if no yellow found
                                self.traci.trafficlight.setPhase(tls, self.tls_phase_map[tls][target_action])
                                self.tls_current_target[tls] = target_action

                            self.last_switch[tls] = self.step_count
                        except Exception:
                            pass
                else:
                    # Target matches current. DON'T call setPhase to avoid timer resets.
                    pass
            except Exception:
                continue

        # GUI visual colors is O(N), but we ONLY do it if render_mode=human
        # so training remains wildly fast.
        if self.render_mode == "human":
            vehicles = self.traci.vehicle.getIDList()
            for v in vehicles:
                try:
                    v_type = self.traci.vehicle.getTypeID(v)
                    if v_type == "ambulance":
                        self.traci.vehicle.setColor(v, (255, 0, 0))
                    elif "bus" in v_type:
                        self.traci.vehicle.setColor(v, (255, 255, 0))
                    elif "truck" in v_type:
                        self.traci.vehicle.setColor(v, (0, 255, 0))
                    elif "bike" in v_type:
                        self.traci.vehicle.setColor(v, (255, 0, 255))
                    else:
                        self.traci.vehicle.setColor(v, (0, 0, 255))
                except Exception:
                    continue

        self.traci.simulationStep()
        self.step_count += 1

        if self.render_mode == "human" and self.step_count % 50 == 0:
            try:
                self.traci.gui.screenshot("View #0", f"screenshots/step_{self.step_count}.png")
            except Exception:
                pass

        next_state = self.get_state()

        try:
            # Separate vehicle and pedestrian waiting times
            total_waiting = 0
            total_queue = 0
            total_ped_waiting = 0
            
            for tls in self.active_tls_list:
                try:
                    lanes = self.tls_lane_map.get(tls, [])
                    v_lanes = [l for l in lanes if "crossing" not in l and "sidewalk" not in l]
                    p_lanes = [l for l in lanes if "crossing" in l or "sidewalk" in l]
                    
                    total_waiting += sum(self.traci.lane.getLastStepHaltingNumber(lane) for lane in v_lanes)
                    total_queue += sum(self.traci.lane.getLastStepVehicleNumber(lane) for lane in v_lanes)
                    total_ped_waiting += sum(self.traci.lane.getLastStepPersonNumber(lane) for lane in p_lanes)
                except Exception:
                    continue
                
        except Exception:
            total_waiting, total_queue, total_ped_waiting = 0, 0, 0

        # Equal priority to vehicle queues and pedestrian waiting times
        reward = -2 * total_waiting - 2 * total_ped_waiting - total_queue

        for v in list(self.active_ambulances):
            try:
                speed = self.traci.vehicle.getSpeed(v)
                # High-priority reward scaling
                reward += 100 if speed > 5 else -500
            except Exception:
                pass

        terminated = self.step_count >= self.max_steps
        truncated = False

        return next_state, reward, terminated, truncated, {}

    def close(self):
        try:
            self.traci.close()
        except Exception:
            pass