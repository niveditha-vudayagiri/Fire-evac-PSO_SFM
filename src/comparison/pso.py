#just a basic pso implementation
import numpy as np
import matplotlib.pyplot as plt

# --- PSO Implementation ---
class PSOParticle:
    def __init__(self, position, velocity, goal):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.best_position = np.copy(self.position)
        self.goal = np.array(goal, dtype=float)
        self.best_distance = np.linalg.norm(self.position - self.goal)
        self.trajectory = [np.copy(self.position)]

    def update_personal_best(self):
        distance = np.linalg.norm(self.position - self.goal)
        if distance < self.best_distance:
            self.best_distance = distance
            self.best_position = np.copy(self.position)

class PSO:
    def __init__(self, num_particles, goal, bounds, w=0.5, c1=1.5, c2=1.5):
        self.particles = []
        self.goal = np.array(goal, dtype=float)
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        for _ in range(num_particles):
            position = np.random.uniform(bounds[0], bounds[1], size=2)
            velocity = np.random.uniform(-1, 1, size=2)
            self.particles.append(PSOParticle(position, velocity, self.goal))
        self.global_best_position = self.particles[0].position
        self.global_best_distance = np.linalg.norm(self.global_best_position - self.goal)
        self.update_global_best()

    def update_global_best(self):
        for p in self.particles:
            if p.best_distance < self.global_best_distance:
                self.global_best_distance = p.best_distance
                self.global_best_position = np.copy(p.best_position)

    def step(self):
        for p in self.particles:
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)
            p.velocity = self.w * p.velocity + cognitive + social
            p.position += p.velocity
            p.position = np.clip(p.position, self.bounds[0], self.bounds[1])
            p.update_personal_best()
            p.trajectory.append(np.copy(p.position))
        self.update_global_best()

    def run(self, max_iters=50):
        for _ in range(max_iters):
            self.step()
        return [p.trajectory for p in self.particles]

# --- SFM Implementation ---
class SFMParticle:
    def __init__(self, position, goal, desired_speed=0.5):
        self.position = np.array(position, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.desired_speed = desired_speed
        self.trajectory = [np.copy(self.position)]

    def step(self, others, dt=1.0, A=2.0, B=1.0):
        # Attractive force toward goal
        direction = self.goal - self.position
        distance = np.linalg.norm(direction)
        if distance > 1e-2:
            direction = direction / distance
        else:
            direction = np.zeros_like(direction)
        force_goal = self.desired_speed * direction

        # Repulsive force from others
        force_rep = np.zeros(2)
        for other in others:
            if other is self:
                continue
            diff = self.position - other.position
            dist = np.linalg.norm(diff)
            if dist < 1e-2:
                continue
            force_rep += A * np.exp(-dist / B) * (diff / dist)

        # Update position
        self.position += (force_goal + force_rep) * dt
        self.trajectory.append(np.copy(self.position))

class SFM:
    def __init__(self, num_particles, goal, bounds):
        self.particles = []
        self.goal = np.array(goal, dtype=float)
        self.bounds = bounds
        for _ in range(num_particles):
            position = np.random.uniform(bounds[0], bounds[1], size=2)
            self.particles.append(SFMParticle(position, self.goal))

    def run(self, max_iters=50):
        for _ in range(max_iters):
            for p in self.particles:
                p.step(self.particles)
                p.position = np.clip(p.position, self.bounds[0], self.bounds[1])
        return [p.trajectory for p in self.particles]

# --- Comparison and Visualization ---
def plot_trajectories(pso_trajs, sfm_trajs, goal):
    plt.figure(figsize=(10, 5))
    for traj in pso_trajs:
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'b--', alpha=0.7, label='PSO' if 'PSO' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(traj[0, 0], traj[0, 1], c='b', marker='o', s=30)
    for traj in sfm_trajs:
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.7, label='SFM' if 'SFM' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(traj[0, 0], traj[0, 1], c='r', marker='x', s=30)
    plt.scatter(goal[0], goal[1], c='g', marker='*', s=200, label='Exit')
    plt.legend()
    plt.title("PSO vs SFM: Agent Trajectories to Exit")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    num_agents = 10
    goal = [8, 8]
    bounds = [0, 10]