
from xmlrpc.client import MAXINT
import random
import heapq
from collections import deque
import pygame
import sys
import matplotlib.pyplot as plt
from pygame.math import clamp

#############################
# Maze (Ship) Generation
#############################
# This function generates a maze (or "ship") as a 2D grid.
# A cell with a value of 1 represents an open area; a cell with 0 represents a wall.
def generate_ship(D):
    """
    Generates a D x D maze representing the ship.
    Starts with a grid full of walls (0's), then "grows" the maze from a random starting cell.
    Cells become open (1) when they are adjacent to exactly one open cell.
    Also, occasionally opens dead ends to create more connectivity.
    """
    # Create a grid with all cells initially blocked (0's)
    grid = [[0 for _ in range(D)] for _ in range(D)]
    # Choose a random starting cell (not on the edge)
    start_i = random.randint(1, D - 2)
    start_j = random.randint(1, D - 2)
    grid[start_i][start_j] = 1  # Mark the starting cell as open

    # "Grow" the maze until no more candidates exist.
    while True:
        candidates = []
        # Loop over interior cells (ignoring border)
        for i in range(1, D - 1):
            for j in range(1, D - 1):
                if grid[i][j] == 0:  # Only consider closed cells
                    open_neighbors = 0
                    # Check all four directions (up, down, left, right)
                    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                        ni, nj = i + di, j + dj
                        if grid[ni][nj] == 1:
                            open_neighbors += 1
                    # Only consider cells that have exactly one open neighbor (helps avoid loops)
                    if open_neighbors == 1:
                        candidates.append((i, j))
        # Stop if no candidate cells remain
        if not candidates:
            break
        # Randomly choose a candidate cell and open it
        cell = random.choice(candidates)
        grid[cell[0]][cell[1]] = 1

    # Optionally open up some dead ends to make the maze less sparse.
    dead_ends = []
    for i in range(D):
        for j in range(D):
            if grid[i][j] == 1:
                open_nbrs = []
                # Count open neighbors for each open cell
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
                        open_nbrs.append((ni, nj))
                # A dead end has only one open neighbor
                if len(open_nbrs) == 1:
                    dead_ends.append((i, j))
    # Randomly open one of the walls adjacent to dead ends to reduce dead-end frequency
    for cell in dead_ends:
        if random.random() < 0.5:
            neighbors = []
            i, j = cell
            for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 0:
                    neighbors.append((ni, nj))
            if neighbors:
                new_cell = random.choice(neighbors)
                grid[new_cell[0]][new_cell[1]] = 1
    return grid

#############################
# Helper Functions
#############################
# These functions assist with pathfinding and distance calculations.

def get_neighbors(pos, grid):
    """
    Given a cell position 'pos' (i, j) and the grid,
    returns a list of adjacent cells (up, down, left, right) that are open (value 1).
    """
    D = len(grid)
    i, j = pos
    nbrs = []
    for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
            nbrs.append((ni, nj))
    return nbrs

def manhattan_distance(a, b):
    """
    Returns the Manhattan distance between two points a and b.
    Manhattan distance is the sum of the absolute differences in the x and y coordinates.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bfs_path(grid, start, goal, obstacles=set()):
    """
    Uses Breadth-First Search (BFS) to find a shortest path from start to goal.
    'obstacles' is a set of positions that the search should avoid.
    Returns a list of cell positions representing the path, or None if no path exists.
    """
    queue = deque([start])
    came_from = {start: None}  # Dictionary to reconstruct the path
    while queue:
        current = queue.popleft()
        # If we reached the goal, break out
        if current == goal:
            break
        # Explore all neighbors of the current cell
        for nbr in get_neighbors(current, grid):
            # Skip neighbors that are obstacles
            if nbr in obstacles:
                continue
            if nbr not in came_from:
                came_from[nbr] = current
                queue.append(nbr)
    if goal not in came_from:
        return None  # No path found
    # Reconstruct the path from goal to start
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path

def multi_source_bfs(fire_set, grid):
    """
    Computes a grid of "fire arrival times" for each open cell.
    It starts from all cells in 'fire_set' (the current fire cells) and uses BFS to
    determine the minimum number of moves it would take for the fire to reach each cell.
    Returns a 2D list (grid) of distances.
    """
    D = len(grid)
    dist = [[float('inf')] * D for _ in range(D)]
    queue = deque()
    # Initialize the distances for cells that are already on fire
    for cell in fire_set:
        i, j = cell
        dist[i][j] = 0
        queue.append(cell)
    # Perform BFS to update distances for all open cells
    while queue:
        i, j = queue.popleft()
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
                if dist[ni][nj] > dist[i][j] + 1:
                    dist[ni][nj] = dist[i][j] + 1
                    queue.append((ni, nj))
    return dist

#############################
# Fire Spread Functions
#############################
# These functions simulate how fire spreads across the grid.

def update_fire(grid, fire_set, q, goal):
    """
    Updates the current fire_set based on the flammability parameter 'q'.
    For each open cell that is not already on fire (and not the goal),
    it calculates the probability of catching fire based on the number of adjacent fire cells.
    The probability is given by: 1 - (1 - q)^K, where K is the number of adjacent fire cells.
    Returns the new fire_set.
    """
    D = len(grid)
    new_fire = set(fire_set)
    for i in range(D):
        for j in range(D):
            if grid[i][j] == 1 and (i, j) not in fire_set:
                K = 0
                for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < D and 0 <= nj < D and (ni, nj) in fire_set:
                        K += 1
                # Calculate probability that this cell catches fire
                prob = 1 - (1 - q) ** K
                # Make sure the goal never catches fire
                if random.random() < prob and (i, j) != goal:
                    new_fire.add((i, j))
    return new_fire

#############################
# Bot Strategies
#############################
# Below are various strategies (bots) to navigate the maze and avoid fire.
# Bot1 follows a fixed precomputed path, Bot2 replans every move, Bot3 avoids fire and adjacent cells,
# and Bot4 is our advanced bot with global planning and adaptive fire escape.

# Bot 1: Fixed precomputed path.
class Bot1:
    def __init__(self, grid, start, button, initial_fire):
        # Precompute a path from the start to the button using BFS (ignoring the fire)
        self.grid = grid
        self.button = button
        self.path = bfs_path(grid, start, button, obstacles={initial_fire})
        if self.path is None:
            self.path = [start]
        self.index = 0  # To keep track of our progress along the path

    def next_move(self, bot_pos, fire_set):
        # Simply follow the fixed path. If we haven't reached the end, take the next step.
        if self.index < len(self.path) - 1:
            self.index += 1
            return self.path[self.index]
        return bot_pos

# Bot 2: Replans every step (avoiding current fire).
class Bot2:
    def __init__(self, grid, button):
        self.grid = grid
        self.button = button

    def next_move(self, bot_pos, fire_set):
        # Compute a new path at every step, avoiding any cell that is on fire.
        path = bfs_path(self.grid, bot_pos, self.button, obstacles=fire_set)
        if path is None or len(path) < 2:
            return bot_pos  # If no safe move, stay in place
        return path[1]  # Take the next step on the path

# Bot 3: Replans every step (avoiding fire and its adjacent cells).
class Bot3:
    def __init__(self, grid, button):
        self.grid = grid
        self.button = button

    def next_move(self, bot_pos, fire_set):
        # First, create a set of cells that are adjacent to fire (these are extra dangerous).
        adj_to_fire = set()
        for cell in fire_set:
            for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                nbr = (cell[0] + di, cell[1] + dj)
                # Only consider valid cells in the grid that are open.
                if 0 <= nbr[0] < len(self.grid) and 0 <= nbr[1] < len(self.grid) and self.grid[nbr[0]][nbr[1]] == 1:
                    adj_to_fire.add(nbr)
        # Combine actual fire cells with cells adjacent to fire.
        obstacles = fire_set.union(adj_to_fire)
        # Try to find a path that avoids both.
        path = bfs_path(self.grid, bot_pos, self.button, obstacles=obstacles)
        if path is None:
            # Fallback: just avoid fire
            path = bfs_path(self.grid, bot_pos, self.button, obstacles=fire_set)
        if path is None or len(path) < 2:
            return bot_pos  # No safe move found
        return path[1]

# Bot 4: Enhanced Global Planning and Adaptive Fire Escape Bot.
# This bot uses a global A* search that factors in:
#   - The number of steps (g_cost)
#   - A penalty for cells where fire is predicted to arrive too soon (using a safety margin)
#   - The Manhattan distance as a heuristic
# It also adapts to the observed fire spread and adds movement inertia to prevent oscillations.
class Bot4:
    def __init__(self, grid, button, safety_threshold=1.0, penalty_weight=10):
        self.grid = grid
        self.button = button
        self.safety_threshold = safety_threshold  # Minimum acceptable safety margin
        self.penalty_weight = penalty_weight      # Penalty weight for low safety margins
        self.prev_fire_set = None  # To track previous fire state and adjust predictions
        self.fire_speed_factor = 1.0  # Adjusts how quickly we expect fire to spread
        self.last_move = None  # To remember the previous move and avoid immediate reversals

    def a_star_path(self, start, goal, fire_distance_grid):
        """
        Performs a global A* search from 'start' to 'goal'.
        The cost function here is:
            g_cost (steps taken + penalty for low safety margin) + h (Manhattan distance to goal)
        The penalty is added if the predicted fire arrival time (from the fire_distance_grid) minus
        the number of steps is less than the safety_threshold.
        Returns the path as a list of positions if found, otherwise returns None.
        """
        open_set = []
        g_cost = {start: 0}
        # h is the Manhattan distance from the start to the goal
        h = manhattan_distance(start, goal)
        f = g_cost[start] + h
        # Push the starting cell onto the priority queue with its cost
        heapq.heappush(open_set, (f, 0, start, [start]))
        closed_set = set()

        while open_set:
            f, steps, pos, path = heapq.heappop(open_set)
            # If we reached the goal, return the path
            if pos == goal:
                return path
            if pos in closed_set:
                continue
            closed_set.add(pos)
            # Explore all neighbors of the current cell
            for nbr in get_neighbors(pos, self.grid):
                new_steps = steps + 1
                i, j = nbr
                # Calculate predicted time for fire to reach this neighbor (using multi-source BFS result)
                predicted_fire_time = fire_distance_grid[i][j] / self.fire_speed_factor
                safe_margin = predicted_fire_time - new_steps  # How safe is this cell?
                penalty = 0
                if safe_margin < self.safety_threshold:
                    # Add a penalty if the safety margin is too low
                    penalty = (self.safety_threshold - safe_margin) * self.penalty_weight
                new_cost = new_steps + penalty  # Total cost so far (g_cost)
                new_h = manhattan_distance(nbr, goal)  # Heuristic cost (h)
                total_cost = new_cost + new_h  # f = g + h
                if nbr not in g_cost or new_cost < g_cost[nbr]:
                    g_cost[nbr] = new_cost
                    heapq.heappush(open_set, (total_cost, new_steps, nbr, path + [nbr]))
        return None  # No path found

    def next_move(self, bot_pos, fire_set):
        """
        Decides the next move for Bot4.
        It updates the fire spread prediction, computes a global path using A*,
        and then returns the next step along that path.
        If no valid path is found, it falls back to choosing the neighbor that minimizes the Manhattan distance.
        """
        # --- Real-Time Fire Spread Adjustment ---
        # Update our fire speed factor based on the change in the fire set.
        if self.prev_fire_set is not None:
            growth = len(fire_set) - len(self.prev_fire_set)
            if growth <= 0:
                self.fire_speed_factor = 0.5  # Fire is spreading slowly => be more aggressive
            elif growth == 1:
                self.fire_speed_factor = 0.7  # Moderate spread
            else:
                self.fire_speed_factor = 1.0  # Fire is spreading fast => play it safe
        else:
            self.fire_speed_factor = 1.0
        # Update previous fire set for the next move comparison
        self.prev_fire_set = fire_set.copy()

        # --- Global Fire Prediction ---
        # Compute a grid of fire arrival times using multi-source BFS (from all current fire cells)
        fire_distance_grid = multi_source_bfs(fire_set, self.grid)

        # --- Global Path Planning (A* Search) ---
        # Compute a full path from our current position to the goal using our custom A* search.
        path = self.a_star_path(bot_pos, self.button, fire_distance_grid)

        if path and len(path) >= 2:
            # If a valid path is found, take the immediate next step on that path.
            next_step = path[1]
            # --- Movement Inertia ---
            # Check if the next step is simply reversing our last move. If so, choose an alternative.
            if self.last_move is not None and next_step == self.last_move:
                # Get all candidate moves excluding the immediate reversal.
                candidates = [c for c in get_neighbors(bot_pos, self.grid) if c != self.last_move]
                if candidates:
                    # Choose the candidate that minimizes Manhattan distance to the goal.
                    next_step = min(candidates, key=lambda c: manhattan_distance(c, self.button))
            # Remember our current position as the last move.
            self.last_move = bot_pos
            return next_step
        else:
            # --- Fallback Strategy ---
            # If the A* search fails (possibly due to heavy fire blocking), fall back to a simple heuristic:
            # Choose the neighbor that minimizes Manhattan distance to the goal.
            candidates = get_neighbors(bot_pos, self.grid)
            if candidates:
                next_step = min(candidates, key=lambda c: manhattan_distance(c, self.button))
                self.last_move = bot_pos
                return next_step
            else:
                # If no moves are possible, stay in place.
                return bot_pos

#############################
# Headless Simulation Function
#############################
# This function runs the simulation without a UI.
# It sets the starting conditions and runs the bot until it reaches the goal or is caught by fire.
def run_simulation_with_conditions(grid, bot_class, q, bot_pos, button, initial_fire, max_steps=1000):
    """
    Runs a simulation on a single generated maze with fixed initial conditions.
    Returns True if the bot reaches the button (goal), or False if it gets caught by fire.
    """
    # Initialize the chosen bot strategy
    if bot_class == Bot1:
        bot = Bot1(grid, bot_pos, button, initial_fire)
    elif bot_class == Bot2:
        bot = Bot2(grid, button)
    elif bot_class == Bot3:
        bot = Bot3(grid, button)
    elif bot_class == Bot4:
        bot = Bot4(grid, button)
    else:
        bot = bot_class(grid, button)
    # Set the initial fire state (as a set of coordinates)
    fire_set = {initial_fire}
    steps = 0
    current_pos = bot_pos
    # Loop until maximum steps reached
    while steps < max_steps:
        new_pos = bot.next_move(current_pos, fire_set)
        current_pos = new_pos
        steps += 1
        # If we reach the button, simulation is successful
        if current_pos == button:
            return True
        # Update the fire spread after each move
        fire_set = update_fire(grid, fire_set, q, button)
        # If the bot moves into a fire cell, it fails
        if current_pos in fire_set:
            return False
    return False  # Return failure if max_steps is reached without success

#############################
# Simple Text Input UI Function
#############################
# This function displays a prompt on the Pygame screen and collects user input.
def get_user_input(screen, prompt):
    """
    Displays a text prompt and collects keyboard input until the user presses Enter.
    Returns the entered text.
    """
    font = pygame.font.SysFont(None, 36)
    input_text = ""
    active = True
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    active = False
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
        screen.fill(BLACK)
        draw_text(screen, prompt, (50, 50), size=36)
        draw_text(screen, input_text, (50, 100), size=36)
        pygame.display.flip()
    return input_text

#############################
# Testing Mode Function
#############################
# This function runs multiple simulations (ships) for each bot and each fire flammability value.
# It then plots the success rate of each bot.
def run_testing_mode():
    """
    In Testing Mode, the user enters the number of simulations to run.
    For each simulation and each flammability value (q), the success of each bot is recorded.
    Finally, a graph is plotted showing the success rate (%) vs. flammability for each bot.
    """
    screen = pygame.display.get_surface()
    input_text = get_user_input(screen, "Enter number of simulations (ships):")
    try:
        N = int(input_text)
    except:
        N = 10  # Default value if input is not valid
    q_values = [round(x * 0.1, 1) for x in range(1, 11)]
    results = {
        "Bot1": {q: [] for q in q_values},
        "Bot2": {q: [] for q in q_values},
        "Bot3": {q: [] for q in q_values},
        "Bot4": {q: [] for q in q_values}
    }
    bot_classes = [Bot1, Bot2, Bot3, Bot4]
    bot_names = ["Bot1", "Bot2", "Bot3", "Bot4"]

    # Run N simulation trials
    for trial in range(N):
        grid = generate_ship(D)
        # Get all open cells in the grid
        open_cells = [(i, j) for i in range(D) for j in range(D) if grid[i][j] == 1]
        if len(open_cells) < 3:
            continue  # Skip this simulation if there are not enough open cells
        # Randomly select the bot start position, the goal (button), and the initial fire cell
        bot_start, button, initial_fire = random.sample(open_cells, 3)
        # Test each flammability value for each bot
        for q_val in q_values:
            for bot_class, name in zip(bot_classes, bot_names):
                state = random.getstate()  # Save the random state for reproducibility
                success = run_simulation_with_conditions(grid, bot_class, q_val, bot_start, button, initial_fire)
                random.setstate(state)  # Restore the state for identical randomness across bots
                results[name][q_val].append(success)
                print(f"Simulation {trial+1}, q={q_val}, {name}: {'Success' if success else 'Failure'}")

    # Calculate the success rate (percentage) for each bot and flammability value.
    final_results = {name: [] for name in bot_names}
    for q_val in q_values:
        for name in bot_names:
            successes = sum(results[name][q_val])
            total = len(results[name][q_val])
            rate = 100 * successes / total if total > 0 else 0
            final_results[name].append(rate)

    # Plot the results using matplotlib.
    plt.figure()
    for name in bot_names:
        plt.plot(q_values, final_results[name], marker='o', label=name)
    plt.xlabel('Flammability (q)')
    plt.ylabel('Success Rate (%)')
    plt.title('Bot Success Rate vs. q (Identical Fire Spread)')
    plt.legend()
    plt.grid(True)
    plt.show()

#############################
# UI / Simulation with Pygame
#############################
# The following functions handle the visual simulation using Pygame.

# Define colors for the UI.
BLACK    = (0, 0, 0)
WHITE    = (255, 255, 255)
GRAY     = (200, 200, 200)
BLUE     = (0, 0, 255)
RED      = (255, 0, 0)
GREEN    = (0, 255, 0)
ORANGE   = (255, 165, 0)

# Global parameters for the simulation.
D = 30             # Dimensions of the grid (D x D)
CELL_SIZE = 20     # Size of each cell in pixels
WIDTH = D * CELL_SIZE  # Overall width of the window
HEIGHT = D * CELL_SIZE + 50  # Overall height (extra space for text)
FPS = 5            # Frames per second for the simulation
q = 1.0            # Default flammability parameter

def draw_grid(screen, grid):
    """
    Draws the grid (maze) on the screen.
    Open cells are drawn in white and walls in black.
    A gray outline is drawn around each cell.
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[i][j] == 1:
                pygame.draw.rect(screen, WHITE, rect)
            else:
                pygame.draw.rect(screen, BLACK, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

def draw_entities(screen, bot_pos, button, fire_set):
    """
    Draws the entities on the screen:
      - The fire is drawn in red.
      - The button (goal) is drawn in green.
      - The bot is drawn as a blue circle.
    """
    # Draw fire cells
    for (i, j) in fire_set:
        rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)
    # Draw the goal (button)
    i, j = button
    rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, rect)
    # Draw the bot as a circle
    i, j = bot_pos
    center = (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
    radius = CELL_SIZE // 2 - 2
    pygame.draw.circle(screen, BLUE, center, radius)

def draw_text(screen, text, pos, color=ORANGE, size=24):
    """
    Renders text on the screen at the given position.
    Used for displaying messages and status information.
    """
    font = pygame.font.SysFont(None, size)
    img = font.render(text, True, color)
    screen.blit(img, pos)

def choose_bot_menu(screen):
    """
    Displays the main menu for the simulation.
    The user can choose one of the bots (1-4) or Testing Mode (option 5).
    """
    screen.fill(BLACK)
    draw_text(screen, "Select Option:", (50, 50), size=36)
    draw_text(screen, "1: Bot1 - Fixed Path (ignores fire spread)", (50, 100))
    draw_text(screen, "2: Bot2 - Replan every step (avoid fire)", (50, 130))
    draw_text(screen, "3: Bot3 - Replan & avoid adjacent fire", (50, 160))
    draw_text(screen, "4: Bot4 - Enhanced Global Planning & Adaptive Escape", (50, 190))
    draw_text(screen, "5: Testing Mode (run multiple simulations)", (50, 220))
    draw_text(screen, "Press 1, 2, 3, 4, or 5", (50, 260))
    pygame.display.flip()

    chosen_option = None
    # Wait for the user to press a key corresponding to an option.
    while chosen_option is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    chosen_option = Bot1
                elif event.key == pygame.K_2:
                    chosen_option = Bot2
                elif event.key == pygame.K_3:
                    chosen_option = Bot3
                elif event.key == pygame.K_4:
                    chosen_option = Bot4
                elif event.key == pygame.K_5:
                    chosen_option = "testing"
    return chosen_option

def run_simulation_ui(grid, bot_class, q):
    """
    Runs a simulation using Pygame for visual output.
    Randomly selects the starting positions for the bot, the button (goal), and the initial fire.
    The simulation runs until the bot either reaches the goal or is caught by fire.
    """
    open_cells = [(i, j) for i in range(D) for j in range(D) if grid[i][j] == 1]
    if len(open_cells) < 3:
        return
    # Randomly sample three distinct open cells: one for the bot, one for the goal, one for the fire.
    bot_pos, button, initial_fire = random.sample(open_cells, 3)
    if bot_class == Bot1:
        bot = Bot1(grid, bot_pos, button, initial_fire)
    elif bot_class == Bot2:
        bot = Bot2(grid, button)
    elif bot_class == Bot3:
        bot = Bot3(grid, button)
    elif bot_class == Bot4:
        bot = Bot4(grid, button)
    else:
        bot = bot_class(grid, button)
    fire_set = {initial_fire}
    steps = 0
    simulation_over = False
    result_text = ""

    clock = pygame.time.Clock()
    screen = pygame.display.get_surface()

    # Main simulation loop
    while not simulation_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get the bot's next move based on its strategy and the current fire state.
        new_bot_pos = bot.next_move(bot_pos, fire_set)
        bot_pos = new_bot_pos
        steps += 1

        # Check if the bot reached the goal (button)
        if bot_pos == button:
            simulation_over = True
            result_text = f"SUCCESS in {steps} steps!"

        # Update the fire spread based on the current flammability parameter q.
        fire_set = update_fire(grid, fire_set, q, button)

        # If the bot moves into a cell that is on fire, it's a failure.
        if bot_pos in fire_set:
            simulation_over = True
            result_text = f"FAILURE in {steps} steps!"

        # Redraw the screen with updated positions.
        screen.fill(BLACK)
        draw_grid(screen, grid)
        draw_entities(screen, bot_pos, button, fire_set)
        draw_text(screen, f"Steps: {steps}", (10, HEIGHT - 40))
        pygame.display.flip()

        clock.tick(FPS)

    # Display final result and wait for user to restart or quit.
    screen.fill(BLACK)
    draw_grid(screen, grid)
    draw_entities(screen, bot_pos, button, fire_set)
    draw_text(screen, result_text, (WIDTH // 4, HEIGHT // 2), size=36)
    draw_text(screen, "Press R to restart or Q to quit", (WIDTH // 4, HEIGHT // 2 + 40))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    waiting = False
                    main()  # Restart simulation
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

def main():
    """
    Main function to initialize Pygame, create the window,
    generate the maze, and start the simulation based on user choice.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("This Bot is on Fire Simulation")
    grid = generate_ship(D)  # Generate a new maze
    chosen_option = choose_bot_menu(screen)  # Let user choose a bot or testing mode
    if chosen_option == "testing":
        run_testing_mode()
    else:
        run_simulation_ui(grid, chosen_option, q)

if __name__ == "__main__":
    main()
