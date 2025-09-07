# Fire Evacuation Simulation with Hybrid PSO-SFM Model

This repository contains the source code for my **novel hybrid evacuation model** that integrates the **Particle Swarm Optimization (PSO)** algorithm with the **Social Force Model (SFM)**.  
The model simulates crowd dynamics and evacuation strategies in multi-floor buildings, focusing on staff coordination policies to improve evacuation efficiency during fire emergencies.

---

## ✨ Features

- **Multi-floor building evacuation** (3 floors, 30m × 30m each).
- **Multiple exits and staircases**:
  - 4 staircases (4m width).
  - 2 main exits on ground floor.
- **Tunable Parameters**:
  - Simulation space
  - Pedestrian count, size, velocity
  - Stairs, doors
  - Opacity
  - Exit queue parameters
  - PSO and SFM parameters
  - Panic mode parameters
  - Iterations/time for simulation
- **Optimization**:
  - PSO integrated with SFM for optimal evacuation paths.
- **Staff coordination policies**:
  - Models the effect of staff guiding evacuees.

---

## Exprimental Base
<img width="300" height="300" alt="Scenario1-normal_layout" src="https://github.com/user-attachments/assets/a20b0ca7-4bfa-4073-93e6-1ff108d27ed2" />

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/niveditha-vudayagiri/Fire-evac-PSO_SFM.git
cd Fire-evac-PSO_SFM
