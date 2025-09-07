import time
import threading
import webbrowser
import numpy as np
import plotly.graph_objects as go
import copy
from simulation.environment import Environment
from simulation.pedestrian import Pedestrian
from simulation.algorithms.random import RandomMover
from simulation.algorithms.pso import PSOMover
from simulation.algorithms.pso_sfm import PSOSFMMover
from simulation.algorithms.mfo_sfm  import MFOSFMMover
from simulation.algorithms.aco_sfm import ACOSFMMover
from simulation.algorithms.sfm import SFMMover
from statistics.statistic_collector import StatisticsCollector
from statistics.dashboard import create_dashboard, live_data

from visualisation.visualizer import VisualizerVPython
from param_config import TIME_STEP, STAFF_RATIO, N_PEDESTRIANS, STAFF_POLICY, MAX_ITERATIONS, SHOW_VISUALS, MIN_STAFF
from statistics.statistic_collector import StatisticsCollector
import random

# Add a lock for live_data
live_data_lock = threading.Lock()

class Simulation:
    def __init__(self, mover_type="random", algorithm_name="random", pedestrians=None, staff_policy=None):
        self.env = Environment()
        self.pedestrians = copy.deepcopy(pedestrians) if pedestrians else self._generate_pedestrians()
        self.mover_type = mover_type
        self.algorithm_name = algorithm_name
        self.staff_policy = staff_policy or STAFF_POLICY
        self.visualizer = VisualizerVPython(
            self.env,
            num_humans=len(self.pedestrians),
            mover_type=self.mover_type,
            algorithm_name=algorithm_name
        )
        self.is_running = False
        if SHOW_VISUALS:
            self.visualizer._setup_scene()
        self.staff_ratio = STAFF_RATIO

    def _generate_pedestrians(self):
        # Ensure at least MIN_STAFF, and distribute staff equally across all floors
        peds = []
        total_peds = N_PEDESTRIANS
        n_staff = max(MIN_STAFF, int(total_peds * STAFF_RATIO))
        staff_indices = set()

        num_floors = 3
        rooms_per_floor = len(self.env.internal_walls)
        base_peds_per_floor = total_peds // num_floors
        extra_peds = total_peds % num_floors
        peds_per_floor = [base_peds_per_floor] * num_floors
        for i in range(extra_peds):
            peds_per_floor[i] += 1  # Distribute remainder

        # Calculate staff per floor (distribute as evenly as possible)
        base_staff_per_floor = n_staff // num_floors
        extra_staff = n_staff % num_floors  # Distribute remainder
        staff_per_floor = [base_staff_per_floor] * num_floors
        for i in range(extra_staff):
            staff_per_floor[i] += 1  # Add extra staff to top floors first

        staff_assigned = 0
        for floor in range(num_floors):
            
            # Step 1: One staff per room using internal wall boundaries (up to staff_per_floor[floor])
            staff_this_floor = 0
            for room_id, bounds in enumerate(self.env.internal_walls):
                if staff_this_floor >= staff_per_floor[floor]:
                    break
                x = np.random.uniform(bounds[0], bounds[2])
                y = np.random.uniform(bounds[1], bounds[3])
                peds.append(Pedestrian(x, y, floor=floor, crowd_id=room_id, is_staff=True))
                staff_assigned += 1
                staff_this_floor += 1
                
            # Step 2: Fill rest of the pedestrians on this floor
            n_peds_on_floor = len([p for p in peds if p.floor == floor])
            peds_needed = peds_per_floor[floor] - n_peds_on_floor
            for _ in range(peds_needed):
                x, y = np.random.uniform(0, 30), np.random.uniform(0, 30)
                peds.append(Pedestrian(x, y, floor=floor, crowd_id=random.randint(0, rooms_per_floor - 1)))

        #TEMP CODE: Generate pedestrians in room 3 of floor 2
        staff_this_floor = 0
        """bounds = self.env.internal_walls[0]  # Room 3 on floor 2
        for _ in range(total_peds):
            x = np.random.uniform(bounds[0], bounds[2])
            y = np.random.uniform(bounds[1], bounds[3])
            peds.append(Pedestrian(x, y, floor=0, crowd_id=3, is_staff=False))"""
 

        """# Temp code : For Exeperimentation
        for _ in range(3):
            x, y = np.random.uniform(0, 30), np.random.uniform(0, 30)
            peds.append(Pedestrian(x, y, floor=2, crowd_id=random.randint(0, rooms_per_floor - 1)))"""

        # Randomly assign remaining staff (from civilians)
        staff_needed = max(0, n_staff - staff_assigned)
        non_staff_candidates = [p for p in peds if not p.is_staff]
        if staff_needed > 0 and len(non_staff_candidates) > 0:
            selected_staff = np.random.choice(non_staff_candidates, min(staff_needed, len(non_staff_candidates)), replace=False)
            for p in selected_staff:
                p.is_staff = True

        # Update colors
        for ped in peds:
            ped.update_color()
            allowed = {'yellow', 'green', 'red', 'blue'}
            if ped.color not in allowed:
                ped.color = 'blue'
        # Defensive: ensure total is correct
        assert len(peds) == N_PEDESTRIANS, f"Generated {len(peds)} peds, expected {N_PEDESTRIANS}"
        return peds

    def run(self, steps=200):
        #Add a small sleep
        mover_kwargs = {}
        if self.mover_type == "random":
            mover = RandomMover(copy.deepcopy(self.pedestrians), self.env)
        elif self.mover_type == "pso":
            mover = PSOMover(copy.deepcopy(self.pedestrians), self.env)
        elif self.mover_type == "sfm":
            mover = SFMMover(copy.deepcopy(self.pedestrians), self.env)
        elif self.mover_type == "pso_sfm":
            mover = PSOSFMMover(copy.deepcopy(self.pedestrians), self.env, staff_ratio=self.staff_ratio,staff_policy=self.staff_policy)
        elif self.mover_type == "mfo_sfm":
            mover = MFOSFMMover(copy.deepcopy(self.pedestrians), self.env)
        elif self.mover_type == "aco_sfm":
            mover = ACOSFMMover(copy.deepcopy(self.pedestrians), self.env)
        else:
            raise ValueError("Unknown mover type")

        # --- Track total per age group at start ---
        algo_key = self.algorithm_name
        if 'age_group_totals' not in live_data[algo_key]:
            age_group_totals = {}
            for p in self.pedestrians:
                age_group = getattr(p, 'age_group', (p.age // 10) * 10 if hasattr(p, 'age') else 'unknown')
                age_group_totals[age_group] = age_group_totals.get(age_group, 0) + 1
            live_data[algo_key]['age_group_totals'] = age_group_totals

        # --- Track total staff vs civilian at start ---
        # Always update staff_type_totals at the start of simulation
        staff_type_totals = {'staff': 0, 'civilian': 0}
        for p in self.pedestrians:
            if getattr(p, 'is_staff', False):
                staff_type_totals['staff'] += 1
            else:
                staff_type_totals['civilian'] += 1
        live_data[algo_key]['staff_type_totals'] = staff_type_totals
        
        simulation_time = 0
        while simulation_time < steps:
            evac_count = mover.move(simulation_time)

            # --------------- TEMP -  Block one of the exit Mid Simulation ---------------
            """if simulation_time == 100 and len(mover.environment.exits)>1:
                mover.environment.exits.pop()  # Block exit at (25, 0)"""

            # -----------------------------------------------------
            # Update visualization
            self.visualizer.update(mover.pedestrians)
            # Map mover_type to algorithm key
            algo_key = self.algorithm_name  # e.g., 'random', 'pso', etc.
            if algo_key not in live_data:
                raise ValueError(f"Unknown mover type: {self.algorithm_name}")
            # Update evacuated and deaths in nested structure
            with live_data_lock:
                live_data[algo_key]['evacuated'].append(evac_count)
                total_peds = len(mover.pedestrians)
                deaths_count = total_peds - evac_count
                live_data[algo_key]['deaths'].append(deaths_count)

                # --- Deaths by floor ---
                # Count deaths per floor at this timestep
                floor_deaths = {}
                for p in mover.pedestrians:
                    if not p.evacuated:
                        floor_deaths[p.floor] = floor_deaths.get(p.floor, 0) + 1
                # Initialize structure if needed
                    # Assume floors 0, 1, 2 (or infer from peds)
                if not live_data[algo_key]['deaths_per_floor']:
                    unique_floors = set(p.floor for p in mover.pedestrians)
                    for floor in unique_floors:
                        live_data[algo_key]['deaths_per_floor'].append({floor: 0})
                live_data[algo_key]['deaths_per_floor'].append(floor_deaths)

                # Track evacuations by age group and staff type
                age_evac = {}
                staff_evac = {'staff': 0, 'civilian': 0}  # Initialize with both keys
                for p in mover.pedestrians:
                    if getattr(p, 'evacuated', False):
                        # Age group tracking
                        age_group = getattr(p, 'age_group', (p.age // 10) * 10 if hasattr(p, 'age') else 'unknown')
                        age_evac[age_group] = age_evac.get(age_group, 0) + 1
                        # Staff vs civilian tracking
                        if getattr(p, 'is_staff', False):
                            staff_evac['staff'] += 1
                        else:
                            staff_evac['civilian'] += 1
                # Debug output
                print(f"[DEBUG] {algo_key} evacuated staff: {staff_evac}")
                if 'evacuated_by_age' not in live_data[algo_key]:
                    live_data[algo_key]['evacuated_by_age'] = []
                live_data[algo_key]['evacuated_by_age'].append(age_evac)
                if 'evacuated_by_staff_type' not in live_data[algo_key]:
                    live_data[algo_key]['evacuated_by_staff_type'] = []
                live_data[algo_key]['evacuated_by_staff_type'].append(staff_evac)
                if len(live_data["timestamps"]) <= simulation_time:
                    live_data["timestamps"].append(simulation_time)
                live_data["current_time"] = simulation_time
            simulation_time += TIME_STEP
        print(f"Simulation Complete for {self.mover_type}!")
        # --- Finalize deaths and evacuated for stats ---
        algo_key = self.algorithm_name
        total_peds = len(mover.pedestrians)
        total_evacuated = sum(1 for p in mover.pedestrians if getattr(p, 'evacuated', False))
        total_deaths = total_peds - total_evacuated
        # Overwrite last value in live_data to ensure consistency
        with live_data_lock:
            if 'evacuated' in live_data[algo_key]:
                live_data[algo_key]['evacuated'][-1] = total_evacuated
            if 'deaths' in live_data[algo_key]:
                live_data[algo_key]['deaths'][-1] = total_deaths
        print(f"[SUMMARY] {algo_key}: Total={total_peds}, Evacuated={total_evacuated}, Deaths={total_deaths}")

if __name__ == "__main__":
    # Generate initial pedestrians
    initial_pedestrians = Simulation()._generate_pedestrians()

    # Initialize simulations with cloned pedestrians and current staff policy
    sim_random = Simulation(mover_type="random", algorithm_name="Random", pedestrians=initial_pedestrians)
    sim_pso = Simulation(mover_type="pso", algorithm_name="PSO", pedestrians=initial_pedestrians)
    sim_sfm = Simulation(mover_type="sfm", algorithm_name="SFM", pedestrians=initial_pedestrians)
    sim_pso_sfm_evac = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Staff all evacuate)", pedestrians=initial_pedestrians, staff_policy="all_evacuate")
    sim_pso_sfm_assist = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Staff all assist)", pedestrians=initial_pedestrians, staff_policy="all_assist")
    sim_pso_sfm_half_assist = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Half Assist)", pedestrians=initial_pedestrians, staff_policy="half_assist_half_leave")
    sim_pso_sfm_assist_mobile = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Assist Mobile)", pedestrians=initial_pedestrians, staff_policy="assist_mobile_first")
    sim_pso_sfm_assist_elderly = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Assist Elderly)", pedestrians=initial_pedestrians, staff_policy="assist_elderly_first")
    sim_pso_sfm_top_evac = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Top Evac)", pedestrians=initial_pedestrians, staff_policy="top_evac")
    sim_pso_sfm_avoid_top = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Avoid Top)", pedestrians=initial_pedestrians, staff_policy="avoid_top")
    sim_pso_sfm_zone_sweep = Simulation(mover_type="pso_sfm", algorithm_name="PSO SFM(Zone Sweep)", pedestrians=initial_pedestrians, staff_policy="zone_sweep")
    sim_mfo_sfm = Simulation(mover_type="mfo_sfm", algorithm_name="MFO SFM", pedestrians=initial_pedestrians)
    sim_aco_sfm = Simulation(mover_type="aco_sfm", algorithm_name="ACO SFM", pedestrians=initial_pedestrians)

    simulations = [
        sim_random,
        sim_pso,
        sim_sfm,
        sim_pso_sfm_evac,
        sim_pso_sfm_assist,
        sim_pso_sfm_half_assist,
        sim_pso_sfm_assist_mobile,
        sim_pso_sfm_assist_elderly,
        sim_pso_sfm_top_evac,
        sim_pso_sfm_avoid_top,
        sim_pso_sfm_zone_sweep,
        sim_mfo_sfm,
        sim_aco_sfm
    ]

    # Start the dashboard first
    app = create_dashboard(simulations)
    server_thread = threading.Thread(target=lambda: app.run(debug=False, port=8050))
    server_thread.daemon = True
    server_thread.start()

    # Wait a moment for dashboard to start
    webbrowser.open('http://localhost:8050')

    # Run each simulation in its own thread
    sim_threads = []
    for sim in simulations:
        thread = threading.Thread(target=sim.run, args=(MAX_ITERATIONS,))
        thread.daemon = True
        thread.start()
        sim_threads.append(thread)
    # Wait for all simulation threads to finish
    for thread in sim_threads:
        thread.join()

    # Keep the main thread alive to allow dashboard and simulation threads to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
        for thread in sim_threads:
            thread.join()
        print("All simulations completed.")
        app.server.shutdown()
        print("Dashboard server shut down.")
        exit(0)