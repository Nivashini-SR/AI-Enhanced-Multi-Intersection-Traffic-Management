# 🚦 AI Enhanced Multi-Intersection Traffic Management
### Transforming Gridlock into Intelligent Green Waves using Deep Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![SUMO](https://img.shields.io/badge/SUMO-Traffic%20Simulator-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-orange)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 Overview

Urban traffic congestion is one of the major challenges faced by modern cities. Conventional fixed-time traffic signals cannot adapt to changing traffic conditions, resulting in increased waiting time, long vehicle queues, and delayed emergency vehicle response.

This project proposes an **AI-based adaptive traffic signal control system** using **Proximal Policy Optimization (PPO)** with **Deep Reinforcement Learning**. The system dynamically adjusts traffic signal phases based on real-time traffic conditions while prioritizing emergency vehicles and minimizing pedestrian waiting time.

The implementation is built using:

- SUMO (Simulation of Urban Mobility)
- TraCI API
- Gymnasium
- Stable-Baselines3
- PyTorch
- Python

# 🎯 Objectives

- Reduce average vehicle waiting time.
- Reduce traffic congestion.
- Minimize queue length.
- Prioritize emergency vehicles.
- Reduce pedestrian waiting time.
- Increase overall traffic throughput.
- Develop a scalable AI-based traffic signal controller.

# ✨ Key Features

- 🚗 Adaptive traffic signal optimization using PPO
- 🚑 Emergency vehicle priority system
- 🚶 Pedestrian-aware signal control
- 🚦 Dynamic green signal allocation
- 🧠 Deep Reinforcement Learning based controller
- 📊 Multi-objective reward function
- ⚡ Parallel training using multiple SUMO environments
- 📈 Performance comparison with Fixed-Time and Actuated controllers

# 🏗️ System Architecture

```
Traffic Demand
      │
      ▼
 SUMO Simulation
      │
      ▼
Gymnasium Environment
      │
      ▼
State Extraction
(Vehicles + Pedestrians + Ambulance)
      │
      ▼
 PPO Agent
      │
      ▼
Select Traffic Signal Phase
      │
      ▼
Traffic Signal Update
      │
      ▼
Reward Calculation
      │
      ▼
Policy Update
      │
      └──────────────► Repeat
```

# 🧠 Reinforcement Learning Framework

## State Space

The observation space consists of **684 features**.

For each of the **114 junctions**, six normalized features are collected:

- Vehicle waiting time (North-South)
- Vehicle waiting time (East-West)
- Vehicle queue length (North-South)
- Vehicle queue length (East-West)
- Pedestrian waiting time
- Distance to nearest ambulance

Observation Dimension:

114 Junctions × 6 Features = 684

## Action Space

Each junction has two possible actions.

| Action | Description |
|---------|-------------|
| 0 | North-South Green |
| 1 | East-West Green |

A **3-second yellow phase** is inserted before switching directions to ensure traffic safety.

## Reward Function

The reinforcement learning agent maximizes the following reward:

R=−2⋅ΣWv−2⋅ΣWp−1⋅ΣQv+Ramb

Where

- **ΣWv** → Total vehicle waiting time
- **ΣWp** → Total pedestrian waiting time
- **ΣQv** → Vehicle queue length
- **Ramb** → Emergency vehicle reward

Emergency reward:

- +100 → Ambulance moving efficiently
- -500 → Ambulance delayed

# 🚑 Emergency Vehicle Priority

Whenever an ambulance enters within **250 meters** of a traffic junction:

- RL control is temporarily overridden
- Green signal is forced toward ambulance direction
- Opposite traffic is stopped
- Route clearing mechanism moves vehicles aside
- Control returns to PPO after ambulance exits

# 🚶 Pedestrian Integration

Unlike many existing systems, pedestrian waiting time is incorporated into the reward function.

Benefits:

- Balanced vehicle and pedestrian movement
- Reduced pedestrian delay
- Improved safety
- Fair traffic management

# ⚙️ Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python 3.9 |
| Traffic Simulator | SUMO |
| API | TraCI |
| Reinforcement Learning | PPO |
| Deep Learning | PyTorch |
| RL Library | Stable-Baselines3 |
| Environment | Gymnasium |
| Configuration | XML |
| Parallel Training | multiprocessing |

# 📂 Project Structure

```
sumo_project/

│── train_rl.py
│── traffic_env.py
│── test_rl_env.py
│── simulation.sumocfg
│── update_sumo_xml.py
│── traffic_model.zip
│── traffic_model_final.zip
│── trips.trips.xml
│── screenshots/
│── README.md
│── requirements.txt

```

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Nivashini-SR/AI-Enhanced-Multi-Intersection-Traffic-Management.git

cd AI-Enhanced-Multi-Intersection-Traffic-Management
```

Create virtual environment

```bash
python -m venv rl_env
```

Activate

Linux

```bash
source rl_env/bin/activate
```

Windows

```bash
rl_env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start the SUMO simulation

```bash
python train_rl.py
```

To test the trained model

```bash
python test_rl_env.py
```

# 📊 Experimental Results

The proposed PPO-based traffic signal control system was evaluated against conventional **Fixed-Time** and **Actuated Traffic Signal Control** methods using a simulated urban traffic network in SUMO.

The results demonstrate significant improvements in traffic efficiency, emergency vehicle response, and pedestrian mobility.

## Performance Improvements

| Metric | Fixed-Time | PPO | Improvement |
|---------|-----------:|----:|------------:|
| Vehicle Wait Time | 67.4 s | 26.8 s | **60.2% ↓** |
| Queue Length | 8.9 vehicles | 3.1 vehicles | **65.1% ↓** |
| Ambulance Delay | 27.3 s | 1.2 s | **95.6% ↓** |
| Pedestrian Wait Time | 5.0 s | 2.1 s | **58.0% ↓** |
| Throughput | 1150 veh/hr | 1420 veh/hr | **23.5% ↑** |

## PPO Training Progress

| Training Steps | Vehicle Wait (s) | Queue Length | Ambulance Delay (s) | Pedestrian Wait (s) |
|---------------:|-----------------:|-------------:|--------------------:|--------------------:|
| 0 (Random) | 67.4 | 8.9 | 27.3 | 5.0 |
| 10,000 | 52.1 | 6.8 | 18.4 | 4.2 |
| 25,000 | 38.4 | 4.5 | 8.2 | 3.5 |
| 50,000 | 26.8 | 3.1 | 1.2 | 2.1 |

The training converged after approximately **40,000 timesteps**, demonstrating stable policy learning and improved traffic management performance. :contentReference[oaicite:0]{index=0}

# 📈 Performance Comparison

### Fixed-Time vs PPO

- Vehicle waiting time reduced by **60.2%**
- Queue length reduced by **65.1%**
- Ambulance delay reduced by **95.6%**
- Pedestrian waiting reduced by **58.0%**
- Traffic throughput increased by **23.5%**

### Actuated vs PPO

| Metric | Actuated | PPO | Improvement |
|---------|----------:|----:|------------:|
| Vehicle Wait Time | 57.3 s | 26.8 s | **53.2% ↓** |
| Queue Length | 7.1 | 3.1 | **56.3% ↓** |
| Ambulance Delay | 20.5 s | 1.2 s | **94.1% ↓** |
| Pedestrian Wait | 4.5 s | 2.1 s | **53.3% ↓** |
| Throughput | 1200 veh/hr | 1420 veh/hr | **18.3% ↑** |

# 🖼️ Screenshots

## Simulation Environment

<img width="832" height="375" alt="image" src="https://github.com/user-attachments/assets/8eb7d709-ce5b-4197-8d4f-e9efc7623602" />

## Emergency Vehicle Priority

<img width="975" height="410" alt="image" src="https://github.com/user-attachments/assets/39ce3c7b-fdae-4eaa-ae22-d3ed5552bea5" />

## Pedestrian Crossing

<img width="975" height="414" alt="image" src="https://github.com/user-attachments/assets/382db475-daac-4fa7-99cc-fc2fce46ba99" />

## Smooth Traffic Flow

<img width="975" height="414" alt="image" src="https://github.com/user-attachments/assets/dd4e2689-4ecb-4881-96c2-6555455cd523" />

## PPO Training Graph

<img width="1400" height="800" alt="image" src="https://github.com/user-attachments/assets/24ed6248-ce89-40d1-9d13-b7df07729d5c" />

## Performance Comparison Graph

<img width="802" height="337" alt="image" src="https://github.com/user-attachments/assets/3b31e75a-8b45-473b-8e90-d0e809bd24b9" />

# 🔬 Methodology

The proposed framework follows a Reinforcement Learning pipeline for adaptive traffic signal optimization.

```

Traffic Demand Generation
          │
          ▼
SUMO Traffic Simulation
          │
          ▼
Gymnasium Environment
          │
          ▼
State Extraction
(Vehicles + Pedestrians + Ambulance)
          │
          ▼
PPO Agent
          │
          ▼
Traffic Signal Decision
          │
          ▼
Reward Calculation
          │
          ▼
Policy Update
          │
          ▼
Repeat Until Convergence

```

### Workflow

1. Generate traffic demand in SUMO.
2. Extract traffic state information.
3. Feed observations to the PPO agent.
4. Predict the optimal traffic signal phase.
5. Execute actions in the SUMO environment.
6. Compute the reward based on traffic conditions.
7. Update the PPO policy.
8. Repeat until convergence.

# 🚀 Future Scope

The following enhancements can further improve the proposed system:

- Vehicle-to-Everything (V2X) communication.
- Real-time traffic data integration.
- Multi-Agent Reinforcement Learning (MARL).
- Deployment on real-world smart city infrastructure.
- Integration of weather and accident information.
- Support for multiple emergency vehicles.
- Transfer learning for different city layouts.
- Edge computing for low-latency traffic control.

# 📚 References

1. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
2. Lopez et al., *Microscopic Traffic Simulation using SUMO*, IEEE ITSC, 2018.
3. Wei et al., *IntelliLight: Reinforcement Learning for Intelligent Traffic Signal Control*, KDD, 2019.
4. Genders & Razavi, *Using Deep Reinforcement Learning for Traffic Signal Control*, 2016.
5. El-Tantawy et al., *MARLIN-ATSC*, IEEE Transactions on ITS, 2013.
6. Brockman et al., *OpenAI Gym*, 2016.
7. Behrisch et al., *SUMO – Simulation of Urban MObility*, 2011.

> *The complete list of references is available in the project report.*

# 👩‍💻 Author

**Nivashini S R**

**B.Sc. Computer Systems and Design**

**PSG College of Technology**

**GitHub:** https://github.com/Nivashini-SR

# ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share your feedback

## 📄 License

This project is intended for **academic and research purposes**.

> **AI Enhanced Multi-Intersection Traffic Management** demonstrates how Deep Reinforcement Learning can transform conventional traffic signal systems into adaptive, intelligent, and emergency-aware traffic management solutions capable of reducing congestion, improving pedestrian safety, and enabling faster emergency response.
