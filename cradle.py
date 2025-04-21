import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime

class NewtonsCradleBenchmark:
    """
    PyMunk implementation of Newton's Cradle benchmark.
    
    This benchmark simulates a Newton's cradle with a configurable number of balls.
    The challenge is to correctly simulate the propagation of forces through multiple
    objects in instantaneous contact.
    """

    def __init__(self, num_balls=5, ball_radius=25.0, ball_mass=1.0, 
                 spacing=1.0, swing_angle=30.0, display=True, 
                 record=False, output_dir="outputs"):
        """
        Initialize the Newton's Cradle benchmark.
        
        Args:
            num_balls: Number of balls in the cradle
            ball_radius: Radius of each ball in pixels
            ball_mass: Mass of each ball
            spacing: Small spacing between balls at rest
            swing_angle: Initial swing angle (in degrees) for the first ball
            display: Whether to display the simulation graphically
            record: Whether to record data for analysis
            output_dir: Directory to save outputs
        """
        self.num_balls = num_balls
        self.ball_radius = ball_radius
        self.ball_mass = ball_mass
        self.spacing = spacing
        self.swing_angle = swing_angle
        self.display = display
        self.record = record
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # PyMunk space
        self.space = None
        
        # Pygame display
        self.screen = None
        self.draw_options = None
        self.width = 800
        self.height = 600
        self.fps = 60
        
        # Objects
        self.balls = []
        self.constraints = []
        
        # Simulation parameters
        self.timestep = 1.0 / self.fps
        
        # State tracking
        self.start_time = None
        self.last_positions = []
        self.energy_log = []
        self.position_log = []
        
    def setup(self):
        """Setup the simulation environment and objects."""
        # Initialize PyMunk space
        self.space = pymunk.Space()
        self.space.gravity = (0, 980)  # Gravity in pixels/s^2
        self.space.damping = 0.999  # Slight damping to prevent infinite oscillation
        
        # Initialize Pygame if display is enabled
        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Newton's Cradle Simulation")
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
        
        # Create the Newton's cradle
        self.create_cradle()
        
        # Setup data collection
        self.start_time = time.time()
        self.last_positions = self.get_ball_positions()
    
    def create_cradle(self):
        """Create the Newton's cradle with all balls and constraints."""
        # Parameters for the cradle
        pivot_height = 150  # Height of the pivot points
        string_length = 200  # Length of the pendulum strings
        
        # Calculate spacing between ball centers when in contact
        ball_separation = 2 * self.ball_radius + self.spacing
        
        # Calculate base positions for all balls in their rest state
        base_y = pivot_height + string_length  # y-coordinate where balls hang
        
        # Calculate leftmost ball x-position to center the cradle
        leftmost_x = self.width / 2 - (self.num_balls - 1) * ball_separation / 2
        
        # Create each ball and its suspension
        for i in range(self.num_balls):
            # Ball position (all aligned in a row)
            x = leftmost_x + i * ball_separation
            
            # Create the ball
            body = pymunk.Body()
            body.position = (x, base_y)
            body.mass = self.ball_mass
            
            # Create ball shape
            shape = pymunk.Circle(body, self.ball_radius)
            shape.elasticity = 0.999  # Very elastic collisions
            shape.friction = 0.0  # Frictionless
            
            # Add to space
            self.space.add(body, shape)
            
            # Create pivot point (static)
            pivot = pymunk.Body(body_type=pymunk.Body.STATIC)
            pivot.position = (x, pivot_height)
            
            # Create constraint (string)
            string = pymunk.PinJoint(pivot, body)
            string.distance = string_length  # Length of the string
            
            # Add constraint to space
            self.space.add(pivot, string)
            
            # Store references
            self.balls.append(body)
            self.constraints.append(string)
        
        # Raise the first ball to the swing angle if required
        if self.num_balls > 0 and self.swing_angle > 0:
            self.swing_first_ball()
    
    def swing_first_ball(self):
        """Swing the first ball to the specified angle."""
        if not self.balls:
            return
            
        # Get the first ball
        ball = self.balls[0]
        
        # Calculate the new position based on swing angle
        angle_rad = np.radians(self.swing_angle)
        
        # Get the pivot position
        pivot_x, pivot_y = self.constraints[0].a.position
        
        # Length of the pendulum (string)
        string_length = self.constraints[0].distance
        
        # Calculate new position
        new_x = pivot_x - string_length * np.sin(angle_rad)
        new_y = pivot_y + string_length * np.cos(angle_rad)
        
        # Set the ball position
        ball.position = (new_x, new_y)
    
    def get_ball_positions(self):
        """Get the current positions of all balls."""
        return [ball.position for ball in self.balls]
    
    def calculate_system_energy(self):
        """Calculate the total energy (kinetic + potential) of the system."""
        total_energy = 0
        
        for i, ball in enumerate(self.balls):
            # Get position and velocity
            pos = ball.position
            vel = ball.velocity
            
            # Get pivot position
            pivot_x, pivot_y = self.constraints[i].a.position
            
            # Calculate potential energy (relative to pivot height)
            # In pymunk, y increases downward so we reverse the height calculation
            mass = self.ball_mass
            g = 980  # Gravity in pixels/s^2
            height = pivot_y - pos[1]  # Height below pivot point (negative)
            potential_energy = mass * g * height
            
            # Calculate kinetic energy
            kinetic_energy = 0.5 * mass * (vel[0]**2 + vel[1]**2)
            
            # Sum up
            total_energy += potential_energy + kinetic_energy
        
        return total_energy
    
    def run(self, num_steps=1000):
        """Run the simulation for the specified number of steps."""
        self.setup()
        
        step_count = 0
        running = True
        
        while running and step_count < num_steps:
            # Handle events if display is enabled
            if self.display:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            
            # Step the simulation
            self.space.step(self.timestep)
            step_count += 1
            
            # Collect data for analysis
            if self.record and step_count % 5 == 0:  # Record every 5th step
                positions = self.get_ball_positions()
                self.position_log.append(positions)
                self.energy_log.append(self.calculate_system_energy())
            
            # Update display if enabled
            if self.display:
                self.screen.fill((255, 255, 255))
                self.space.debug_draw(self.draw_options)
                pygame.display.flip()
                self.clock.tick(self.fps)
            
            # Print progress
            if step_count % 100 == 0:
                print(f"Step {step_count}/{num_steps}")
        
        # Save data if recording was enabled
        if self.record:
            self.save_data()
        
        # Cleanup Pygame if it was initialized
        if self.display:
            pygame.quit()
    
    def save_data(self):
        """Save the collected data for analysis."""
        # Convert logs to numpy arrays
        positions = np.array(self.position_log)
        energy = np.array(self.energy_log)
        
        # Save raw data
        np.save(os.path.join(self.output_dir, "positions.npy"), positions)
        np.save(os.path.join(self.output_dir, "energy.npy"), energy)
        
        # Create plots
        self.create_plots()
        
        print(f"Data saved to {self.output_dir}")
    
    def create_plots(self):
        """Create visualization plots from the recorded data."""
        # Skip if there's no data
        if not self.position_log:
            return
            
        # Create plot of ball x-positions over time
        positions = np.array(self.position_log)
        energy = np.array(self.energy_log)
        
        # Create x-position plot
        plt.figure(figsize=(10, 6))
        for i in range(self.num_balls):
            plt.plot(positions[:, i, 0], label=f"Ball {i+1}")
        
        plt.xlabel("Simulation Step")
        plt.ylabel("X Position (pixels)")
        plt.title("Newton's Cradle Ball Positions")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "ball_positions.png"))
        plt.close()
        
        # Create energy plot
        plt.figure(figsize=(10, 6))
        plt.plot(energy)
        plt.xlabel("Simulation Step")
        plt.ylabel("Energy")
        plt.title("Total System Energy")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "energy.png"))
        plt.close()

def main():
    """Main function to run the Newton's Cradle benchmark."""
    parser = argparse.ArgumentParser(description="Newton's Cradle Benchmark in PyMunk")
    parser.add_argument("--num_balls", type=int, default=5, help="Number of balls")
    parser.add_argument("--ball_radius", type=float, default=25.0, help="Radius of each ball in pixels")
    parser.add_argument("--ball_mass", type=float, default=1.0, help="Mass of each ball")
    parser.add_argument("--spacing", type=float, default=1.0, help="Spacing between balls")
    parser.add_argument("--swing_angle", type=float, default=30.0, help="Initial swing angle in degrees")
    parser.add_argument("--no_display", action="store_true", help="Run without display")
    parser.add_argument("--record", action="store_true", help="Record simulation data")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("--output_dir", type=str, default="outputs/newtons_cradle", help="Output directory")
    
    args = parser.parse_args()
    
    benchmark = NewtonsCradleBenchmark(
        num_balls=args.num_balls,
        ball_radius=args.ball_radius,
        ball_mass=args.ball_mass,
        spacing=args.spacing,
        swing_angle=args.swing_angle,
        display=not args.no_display,
        record=args.record,
        output_dir=args.output_dir
    )
    
    benchmark.run(num_steps=args.num_steps)

if __name__ == "__main__":
    main()