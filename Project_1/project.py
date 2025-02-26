
from xmlrpc.client import MAXINT
from operator import le
from pickle import FALSE

import random
import heapq
from collections import deque
import pygame
import sys

from pygame.math import clamp

#############################
# Maze (Ship) Generation
#############################

def generate_ship(D):
    """
    Generates a D x D maze representing the ship.
    Cells with 1 are open and cells with 0 are blocked.

    Algorithm:
      1. Start with a grid full of blocked cells.
      2. Pick a random interior cell and open it.
      3. Iteratively, find all blocked cells that have exactly one open neighbor.
         Randomly pick one such cell and open it.
      4. Then, for each dead-end (open cell with one open neighbor), with 50% probability,
         open one of its closed neighbors.
    """
    grid = [[0 for _ in range(D)] for _ in range(D)]
    # Pick a random interior cell (avoid the border)
    start_i = random.randint(1, D - 2)
    start_j = random.randint(1, D - 2)
    grid[start_i][start_j] = 1

    # Iteratively open blocked cells with exactly one open neighbor.
    while True:
        candidates = []
        for i in range(1, D - 1):
            for j in range(1, D - 1):
                if grid[i][j] == 0:
                    open_neighbors = 0
                    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        ni, nj = i + di, j + dj
                        if grid[ni][nj] == 1:
                            open_neighbors += 1
                    if open_neighbors == 1:
                        candidates.append((i, j))
        if not candidates:
            break
        cell = random.choice(candidates)
        grid[cell[0]][cell[1]] = 1

    # Open some dead ends: for each open dead end (only one open neighbor),
    # with 50% probability, open one of its closed neighbors.
    dead_ends = []
    for i in range(D):
        for j in range(D):
            if grid[i][j] == 1:
                open_nbrs = []
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
                        open_nbrs.append((ni, nj))
                if len(open_nbrs) == 1:
                    dead_ends.append((i, j))
    for cell in dead_ends:
        if random.random() < 0.5:
            neighbors = []
            i, j = cell
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 0:
                    neighbors.append((ni, nj))
            if neighbors:
                new_cell = random.choice(neighbors)
                grid[new_cell[0]][new_cell[1]] = 1
    return grid

def get_neighbors(pos, grid):
    """Return adjacent open neighbors (up/down/left/right) for a given position."""
    D = len(grid)
    i, j = pos
    nbrs = []
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < D and 0 <= nj < D and grid[ni][nj] == 1:
            nbrs.append((ni, nj))
    return nbrs

#############################
# Path Planning Algorithms
#############################

def bfs_path(grid, start, goal, obstacles=set()):
    """
    Uses Breadth-First Search (BFS) to find a shortest path from start to goal.
    'obstacles' is a set of positions that cannot be traversed.
    Returns a list of positions (from start to goal) if a path is found, else None.
    """
    queue = deque([start])
    came_from = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nbr in get_neighbors(current, grid):
            if nbr in obstacles:
                continue
            if nbr not in came_from:
                came_from[nbr] = current
                queue.append(nbr)
    if goal not in came_from:
        return None
    # Reconstruct path
    # return goal_found(came_from,goal)
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path

def every_path_search(grid, start, goal):
    """
    For each neighbor of the goal, run BFS to find the shortest path from start to that neighbor,
    then append the goal to complete the path. This returns one candidate path per neighbor.
    """
    paths = []
    paths_max = []

    # Get neighbors of the goal (the button)
    goal_neighbors = get_neighbors(goal, grid)
    if not goal_neighbors:
        return paths, paths_max

    # For each neighbor, compute the BFS shortest path
    for nbr in goal_neighbors:
        path = bfs_path(grid, start, nbr)
        if path is not None:
            full_path = path + [goal]
            paths.append(full_path)
            paths_max.append(len(full_path))

    return paths, paths_max

def best_path(paths,path_max, fire_paths, fire_max,q):
    # print("best_path_started")
    best_path=0
    fastest=MAXINT
    path_info=[]
    count=0
    for i in range (0,len(paths)):
        for j in range(0,len(fire_paths)):
            if (paths[i][path_max[i]-1]==fire_paths[j][fire_max[j]-1]):
                path_info.append([i,j])
                count+=1
    bp_found=False
    for i,j in path_info:
        if path_max[i] < fire_max[j]:
            bp_found=True
            if path_max[i]<fastest:
                fastest=path_max[i]
                best_path=i
    if not bp_found:
        modify_q=q+0.2
        while (not bp_found):
            # print("in while")
            for i,j in path_info:
                # print("in for loop")
                # print(f" {path_max[i]*modify_q}  <    {fire_max[j]}")
                if path_max[i]*modify_q < (fire_max[j]):
                    bp_found=True
                    if path_max[i]<fastest:
                        fastest=path_max[i]
                        best_path=i
            if modify_q<0:
                return
            modify_q-=0.1
            # print(f"stuck here in best_path  {modify_q}")
    return paths[best_path]




def goal_found(came_from,last_found,goal,nbr,curr):
    cur=goal
    path=[]
    count=0
    came_from[nbr]=curr
    came_from[last_found]=goal
    print(f"goal is {goal} last_found is {last_found}")
    print(came_from)
    while cur is not None:
        path.append(cur)
        # print(cur)
        cur=came_from[cur]
        count+=1
    path.reverse()
    return path, count


#############################
# Fire Spreading Function
#############################

def update_fire(grid, fire_set, q, goal):
    """
    Updates the set of burning cells (fire_set) simultaneously.
    For each open cell that is not burning, count K = number of burning neighbors.
    That cell catches fire with probability 1 - (1 - q)^K.
    Button can't catch fire
    """
    D = len(grid)
    new_fire = set(fire_set)  # start with cells already burning
    for i in range(D):
        for j in range(D):
            if grid[i][j] == 1 and (i, j) not in fire_set:
                K = 0
                for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < D and 0 <= nj < D and (ni, nj) in fire_set:
                        K += 1
                # Compute probability to catch fire
                prob = 1 - (1 - q) ** K
                if (random.random() < prob and goal!=(i,j)):
                    new_fire.add((i, j))
    return new_fire

#############################
# Bot Strategies
#############################

# Bot 1: Plans the entire path (ignoring subsequent fire spread) once at t=0.
class Bot1:
    def __init__(self, grid, start, button, initial_fire):
        self.grid = grid
        self.button = button
        # For planning, treat the initial fire cell as blocked.
        self.path = bfs_path(grid, start, button, obstacles={initial_fire})
        if self.path is None:
            # If no path is found, the bot will simply stay in place.
            self.path = [start]
        self.index = 0

    def next_move(self, bot_pos, fire_set):
        # Follow the precomputed path regardless of fire spread.
        if self.index < len(self.path) - 1:
            self.index += 1
            return self.path[self.index]
        return bot_pos  # no move if already at the end

# Bot 2: Re-plans at every time step, avoiding current fire cells.
class Bot2:
    def __init__(self, grid, button):
        self.grid = grid
        self.button = button

    def next_move(self, bot_pos, fire_set):
        path = bfs_path(self.grid, bot_pos, self.button, obstacles=fire_set)
        if path is None or len(path) < 2:
            return bot_pos  # no valid move found; stay in place
        return path[1]

# Bot 3: Re-plans at every time step, trying first to avoid both fire and cells adjacent to fire.
class Bot3:
    def __init__(self, grid, button):
        self.grid = grid
        self.button = button

    def next_move(self, bot_pos, fire_set):
        # Compute cells adjacent to fire.
        adj_to_fire = set()
        for cell in fire_set:
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nbr = (cell[0] + di, cell[1] + dj)
                if (0 <= nbr[0] < len(self.grid) and 0 <= nbr[1] < len(self.grid) and
                    self.grid[nbr[0]][nbr[1]] == 1):
                    adj_to_fire.add(nbr)
        obstacles = fire_set.union(adj_to_fire)
        path = bfs_path(self.grid, bot_pos, self.button, obstacles=obstacles)
        if path is None:
            # Fall back to planning that avoids only fire cells.
            path = bfs_path(self.grid, bot_pos, self.button, obstacles=fire_set)
        if path is None or len(path) < 2:
            return bot_pos
        return path[1]

# Bot 4: A custom bot that uses A* search with a cost that adds a risk penalty
# for being adjacent to fire.
# still trying to figure out exatly what to for this brainstorming
#
class Bot4:
    def __init__(self, grid, button, start, fire_start,q):
        self.grid = grid
        self.button = button
        self.index= 0
        paths,path_max =every_path_search(grid, start, button)
        # print(f" these are the paths{paths}")
        # print(f" {path_max}")
        fire_paths, fire_max = every_path_search(grid, fire_start, button)
        self.path=best_path(paths,path_max, fire_paths, fire_max,q)

    def next_move(self, bot_pos, fire_set):
        # Follow the precomputed path regardless of fire spread.
        if self.index < len(self.path) - 1:
            self.index += 1
            return self.path[self.index]
        return bot_pos  # no move if already at the end

#############################
# UI / Simulation with Pygame
#############################

# Define colors
BLACK    = (0, 0, 0)
WHITE    = (255, 255, 255)
GRAY     = (200, 200, 200)
BLUE     = (0, 0, 255)
RED      = (255, 0, 0)
GREEN    = (0, 255, 0)
ORANGE   = (255, 165, 0)

# Global parameters for the grid and simulation
D = 30             # Grid dimensions (D x D)
CELL_SIZE = 20     # Size of each cell in pixels
WIDTH = D * CELL_SIZE
HEIGHT = D * CELL_SIZE + 50  # extra space for text/status
FPS = 5           # Simulation frames per second
q = 1.0           # Flammability parameter

def draw_grid(screen, grid):
    """Draws the ship grid."""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[i][j] == 1:
                pygame.draw.rect(screen, WHITE, rect)
            else:
                pygame.draw.rect(screen, BLACK, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)  # grid lines

def draw_entities(screen, bot_pos, button, fire_set):
    """Draws the bot, the button, and the fire cells."""
    # Draw fire cells (red)
    for (i, j) in fire_set:
        rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)
    # Draw button cell (green)
    i, j = button
    rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, GREEN, rect)
    # Draw bot (blue circle)
    i, j = bot_pos
    center = (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2)
    radius = CELL_SIZE // 2 - 2
    pygame.draw.circle(screen, BLUE, center, radius)

def draw_text(screen, text, pos, color=ORANGE, size=24):
    """Renders text on the screen."""
    font = pygame.font.SysFont(None, size)
    img = font.render(text, True, color)
    screen.blit(img, pos)

def choose_bot_menu(screen):
    """Displays a menu to select a bot strategy."""
    screen.fill(BLACK)
    draw_text(screen, "Select Bot Strategy:", (50, 50), size=36)
    draw_text(screen, "1: Bot1 - Fixed Path (ignores fire spread)", (50, 100))
    draw_text(screen, "2: Bot2 - Replan every step (avoid fire)", (50, 130))
    draw_text(screen, "3: Bot3 - Replan & avoid adjacent fire", (50, 160))
    draw_text(screen, "4: Bot4 - Path with highest chance of success", (50, 190))
    draw_text(screen, "Press 1, 2, 3, or 4 to select", (50, 240))
    pygame.display.flip()

    chosen_bot = None
    while chosen_bot is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    chosen_bot = Bot1
                elif event.key == pygame.K_2:
                    chosen_bot = Bot2
                elif event.key == pygame.K_3:
                    chosen_bot = Bot3
                elif event.key == pygame.K_4:
                    chosen_bot = Bot4
    return chosen_bot

def run_simulation_ui(grid, bot_class, q):
    """
    Runs the simulation with a UI.
    Places the bot, the button, and the initial fire in random open cells.
    Then updates the simulation step-by-step while drawing the state.
    """
    open_cells = [(i, j) for i in range(D) for j in range(D) if grid[i][j] == 1]
    if len(open_cells) < 3:
        print("Not enough open cells to start simulation.")
        return

    bot_pos, button, initial_fire = random.sample(open_cells, 3)
    # Instantiate the bot (Bot1 requires the start and initial fire, others just need grid and button)
    if bot_class == Bot1:
        bot = Bot1(grid, bot_pos, button, initial_fire)
    elif bot_class == Bot4:
        print("ran")
        bot = Bot4(grid,button,bot_pos,initial_fire,q)
    else:
        bot = bot_class(grid, button)

    fire_set = {initial_fire}
    steps = 0
    simulation_over = False
    result_text = ""

    clock = pygame.time.Clock()
    screen = pygame.display.get_surface()

    while not simulation_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Bot takes a move.
        new_bot_pos = bot.next_move(bot_pos, fire_set)
        bot_pos = new_bot_pos
        steps += 1

        # Check if bot reaches the button.
        if bot_pos == button:
            simulation_over = True
            result_text = f"SUCCESS in {steps} steps!"

        # Spread the fire.
        fire_set = update_fire(grid, fire_set, q, button)

        # Check if bot is on a burning cell.
        if bot_pos in fire_set:
            simulation_over = True
            result_text = f"FAILURE in {steps} steps!"

        # Draw the current simulation state.
        screen.fill(BLACK)
        draw_grid(screen, grid)
        draw_entities(screen, bot_pos, button, fire_set)
        draw_text(screen, f"Steps: {steps}", (10, HEIGHT - 40))
        pygame.display.flip()

        clock.tick(FPS)

    # Simulation ended: display result message.
    screen.fill(BLACK)
    draw_grid(screen, grid)
    draw_entities(screen, bot_pos, button, fire_set)
    draw_text(screen, result_text, (WIDTH // 4, HEIGHT // 2), size=36)
    draw_text(screen, "Press R to restart or Q to quit", (WIDTH // 4, HEIGHT // 2 + 40))
    pygame.display.flip()

    # Wait for user input.
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    waiting = False
                    main()  # Restart the simulation.
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("This Bot is on Fire Simulation")

    # Generate the ship (maze)
    grid = generate_ship(D)

    # Show bot selection menu.
    chosen_bot = choose_bot_menu(screen)

    # Run the simulation UI with the chosen bot.
    run_simulation_ui(grid, chosen_bot, q)

if __name__ == "__main__":
    main()
