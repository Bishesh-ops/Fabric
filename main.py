# The Fabric is represented as a 2D grid of points connected by springs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class Node:
    def __init__(self, position, mass=1.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)
        self.mass = mass

    def apply_force(self, force):
        self.force += np.array(force, dtype=float)

    def reset_force(self):
        self.force.fill(0)

    def update(self, dt, damping):
        acceleration = self.force / self.mass
        self.velocity += acceleration * dt
        self.velocity *= (1 - damping)
        self.position += self.velocity * dt


class Spring:
    def __init__(self, node1, node2, rest_length, stiffness):
        self.node1 = node1
        self.node2 = node2
        self.rest_length = rest_length
        self.stiffness = stiffness

    def apply_force(self):
        delta = self.node2.position - self.node1.position
        distance = np.linalg.norm(delta)
        direction = delta / (distance - 1e-6)
        displacement = distance - self.rest_length
        force = self.stiffness * displacement * direction

        self.node1.apply_force(force)
        self.node2.apply_force(-force)


class Fabric:
    def __init__(self, grid_size, node_spacing, stiffness):
        self.grid_size = grid_size
        self.node_spacing = node_spacing
        self.nodes = []
        self.springs = []
        self.stiffness = stiffness
        self.create_grid()

    def create_grid(self):
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                position = [i * self.node_spacing, j * self.node_spacing, 0]
                node = Node(position)
                row.append(node)
            self.nodes.append(row)
        # Creating the springs
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i < self.grid_size - 1:  # Vertical springs
                    self.springs.append(Spring(self.nodes[i][j], self.nodes[i + 1][j], self.node_spacing, self.stiffness))
                if j < self.grid_size - 1: # Horizontal springs
                    self.springs.append(Spring(self.nodes[i][j], self.nodes[i][j + 1], self.node_spacing, self.stiffness))

    def apply_weight(self, position, weight, radius):
        for row in self.nodes:
            for node in row:
                dist = np.linalg.norm(node.position[:2] - position[:2])
                if dist < radius:
                    node.apply_force([0, 0, -weight / (1 + dist)])

    def check_collision(self, node, ground_level=-0.08):
        """Handle collision with the ground."""
        if node.position[2] < ground_level:
            node.position[2] = ground_level
            node.velocity[2] *= -0.5  # Reflect the velocity with damping

    def reset_forces(self):
        for row in self.nodes:
            for node in row:
                node.reset_force()

    def update(self, dt, damping):
        for spring in self.springs:
            spring.apply_force()
        for row in self.nodes:
            for node in row:
                node.update(dt, damping)
                self.check_collision(node)

    def get_positions(self):
        return np.array([[node.position for node in row] for row in self.nodes])


class Simulation:
    def __init__(self, fabric, dt, damping):
        self.fabric = fabric
        self.dt = dt
        self.damping = damping
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def run(self, steps, weight_position, weight, radius):        # Animation
        def update_frame(frame):
            self.fabric.reset_forces()
            self.fabric.apply_weight(weight_position, weight, radius)
            self.fabric.update(self.dt, self.damping)
            positions = self.fabric.get_positions()

            self.ax.clear()
            x = positions[:, :, 0]
            y = positions[:, :, 1]
            z = positions[:, :, 2]
            self.ax.plot_surface(x, y, z, color='blue', rstride=1, cstride=1, alpha=0.5)

        ani = FuncAnimation(self.fig, update_frame, frames=steps, interval=50, repeat=False)

        plt.show()


if __name__ == '__main__':
    # Parameters
    grid_size = 20
    node_spacing = 1.0
    stiffness = 400.0
    dt = 0.01
    damping = 0.1
    weight = 1.0
    radius = 2.0
    weight_position = [10.0, 10.0, 0.0] # Center of the grid

    fabric = Fabric(grid_size, node_spacing, stiffness)
    simulation = Simulation(fabric, dt, damping)

    simulation.run(steps=200, weight_position=weight_position, weight=weight, radius=radius)

