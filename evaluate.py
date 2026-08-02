import traci
import numpy as np

def measure_waiting():
    total_wait = 0

    lanes = traci.lane.getIDList()

    for lane in lanes:
        total_wait += traci.lane.getWaitingTime(lane)

    return total_wait