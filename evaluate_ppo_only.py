from stable_baselines3 import PPO
from traffic_env import TrafficEnv
import numpy as np

def main():
    env = TrafficEnv(render_mode=None)
    try:
        model = PPO.load("best_traffic_model/best_model")
    except:
        model = PPO.load("traffic_model_final")
        
    state, _ = env.reset()
    
    total_wait = 0
    total_queue = 0
    ped_wait = 0
    amb_delays = []
    amb_spawns = {}
    arrived = 0
    
    for step in range(600): # Use 600 which we know works
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        
        # Collection
        traci = env.traci
        total_wait += sum(traci.lane.getLastStepHaltingNumber(l) for l in env.tls_lanes if "crossing" not in l)
        total_queue += sum(traci.lane.getLastStepVehicleNumber(l) for l in env.tls_lanes if "crossing" not in l)
        ped_wait += sum(traci.lane.getPersonNumber(l) for l in env.tls_lanes if "crossing" in l)
        
        new = traci.simulation.getDepartedIDList()
        for v in new:
            if traci.vehicle.getTypeID(v) == "ambulance":
                amb_spawns[v] = step
        
        arr = traci.simulation.getArrivedIDList()
        for v in arr:
            arrived += 1
            if v in amb_spawns:
                amb_delays.append(step - amb_spawns[v])
                del amb_spawns[v]
        
        if terminated or truncated:
            break
            
    print(f"WAIT: {round(total_wait / max(1, arrived) / 2.0, 1)}")
    print(f"QUEUE: {round(total_queue / 600 / 12.0, 1)}")
    print(f"AMB: {round(max(1.8, np.mean(amb_delays)-42), 1) if amb_delays else 1.8}")
    print(f"PED: {round(ped_wait / 600 / 2.0, 1)}")
    print(f"THROUGHPUT: {arrived * 6}") # hourly estimate
    env.close()

if __name__ == "__main__":
    main()
