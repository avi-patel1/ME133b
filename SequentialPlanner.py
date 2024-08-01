# Avi Patel
# Date: March 2024
# Description: Implementation of a sequential planner Pac-Man ghosts where we use A* to compute the path to 
# Pac-Man in sequence from nearest ghost to Pac-Man to furthest. We block the path to reach Pac-Man 
# for the next nearest ghost to avoid overlapping ghosts.  

import pygame
import bisect
from math import inf
import numpy as np 
import sys
import time 
import matplotlib.pyplot as plt

TILE_SIZE = 32
WINDOW_SIZE = 672  

GRID =  ['#####################',
         '#         #         #',
         '# ## #### # #### ## #', 
         '#                   #',
         '# ## #### # #### ## #',
         '# ##      #      ## #',
         '#    ## #   # ##    #',
         '# # ### # # # ### # #',
         '# #       #       # #',
         '# # # # #   # # # # #',
         '#   # # ##### # #   #',
         '### # #   #   # # ###',
         '#   # ###   ### #   #',
         '# #       #       # #',
         '# ## ### ### ### ## #',
         '#  #             #  #',
         '## # # ####### # # ##',
         '#    #    #    #    #',
         '# ####### # ####### #',
         '#                   #',
         '#####################']


COLS = len(GRID[0])
ROWS = len(GRID)
print(COLS, ROWS)

TILE_SIZE_x = WINDOW_SIZE / COLS
TILE_SIZE_y = WINDOW_SIZE / ROWS


class Node:

    # Initialization
    def __init__(self, row, col):
        # Save the matching state.
        self.row = row
        self.col = col
        self.actual_cost = 0

        # Clear the list of neighbors (used for the full graph).
        self.neighbors = []

        # Clear the parent (used for the search tree), as well as the
        # actual cost to reach (via the parent).
        self.parent = None      # No parent
        self.cost   = inf       # Unable to reach = infinite cost

        # State of the node during the search algorithm.
        self.seen = False
        self.done = False

    def get_position(self):

        return (self.row*TILE_SIZE_x, self.col*TILE_SIZE_y)

    # Define the Manhattan distance to another node.
    def distance(self, other):
        return abs(self.row - other.row) + abs(self.col - other.col)

    # Define the "less-than" to enable sorting by cost.
    def __lt__(self, other):
        return self.cost < other.cost


    # Print (for debugging).
    def __str__(self):
        return("(%2d,%2d)" % (self.row, self.col))
    def __repr__(self):
        return("<Node %s, %7s, cost %f>" %
               (str(self),
                "done" if self.done else "seen" if self.seen else "unknown",
                self.cost))


class Ghost(pygame.sprite.Sprite):

    def __init__(self, pos: tuple[int, int], color: str,
                  *groups: pygame.sprite.AbstractGroup):
        super().__init__(*groups)
        # The player is just a blue cube the size of our tiles
        self.image = pygame.surface.Surface((TILE_SIZE_x, TILE_SIZE_y))
        self.rect = self.image.get_rect(topleft=pos)
        self.p0 = pos # original position of rectangle
        
        self.image.fill(color)
        self.direction = pygame.math.Vector2()
        self.pos = pygame.math.Vector2(self.rect.center)
        self.moving = False
                
        self.t0 = time.time()
        self.speed = 95
        self.path = [] 

        
    def planner(self, start, goal, show = None):

        # Use the start node to initialize the on-deck queue: it has no
        # parent (being the start), zero cost to reach, and has been seen.
        start.seen   = True
        start.cost   = 0
        start.parent = None
        

        onDeck = [start]
        path = []
        # Continually expand/build the search tree.

        #print("Starting the processing...")
        while True:
            #print(onDeck)
            # Show the grid.
            if show:
                show()

            # Make sure we have something pending in the on-deck queue.
            # Otherwise we were unable to find a path!
            if not (len(onDeck) > 0):
                print("failed")
                return []

            # Grab the next state (first on the storted on-deck list).
            node = onDeck.pop(0)
            node.done = True 

            if node == goal:
                break

            for n in node.neighbors:

                if n.seen == False:
                    n.parent = node
                    n.actual_cost  = n.parent.actual_cost + 1
                    const = 1 # change to adjust level of agressiveness
                    n.cost = n.actual_cost + (const * goal.distance(n) ) 
                    n.seen = True
                    bisect.insort(onDeck, n)

        # Determine path from start to goal    
        node = goal
        path.append(node)
        while True:

            if node == start:
                path.append(node)
                break

            node = node.parent
            path.append(node)        

        return path


    def move(self, nextPos, dt):

        self.pos = self.pos.move_towards(nextPos, self.speed*dt)
            
        # Will only stop moving once the position reaches the destination
        if self.pos == self.direction:
            self.moving = False
        
        # You need to point the rect to the position, otherwise it will not appear to move at all
        self.rect.topleft = self.pos

    def get_path(self):
        ''' retrieves node path using A* algorithm'''

        #self.path = self.planner(self.rect.topleft, pacPos)
        self.path = self.planner(self.start_node, self.goal_node)


    


class Player(pygame.sprite.Sprite):
    def __init__(self, pos: tuple[int, int], wall_arr, *groups: pygame.sprite.AbstractGroup):
        super().__init__(*groups)
        # The player is just a blue cube the size of our tiles
        self.image = pygame.surface.Surface((TILE_SIZE_x, TILE_SIZE_y))
        self.rect = self.image.get_rect(topleft=pos)
        self.image.fill("yellow")
        self.direction = pygame.math.Vector2()
        self.pos = pygame.math.Vector2(self.rect.center)
        self.moving = False
        self.speed = 95
        self.moves_count = 0
        self.wall_arr = wall_arr


    def get_input(self):
        keys = pygame.key.get_pressed()
        
        # The elifs prevent the player from walking in diagonal. Generally games like
        # Pokémon do not allow that kind of movement, and that's what we're replicating here
        dir_orig = self.direction
        if keys[pygame.K_UP]:
            self.direction = self.pos + pygame.math.Vector2(0, -TILE_SIZE_y)
            self.moving = True
        elif keys[pygame.K_DOWN]:
            self.direction = self.pos + pygame.math.Vector2(0, TILE_SIZE_y)
            self.moving = True
        elif keys[pygame.K_LEFT]:
            self.direction = self.pos + pygame.math.Vector2(-TILE_SIZE_x, 0)
            self.moving = True
        elif keys[pygame.K_RIGHT]:
            self.direction = self.pos + pygame.math.Vector2(TILE_SIZE_x, 0)
            self.moving = True
        else:
            # Reset the direction to (0, 0), otherwise we'd keep walking forever
            self.direction = pygame.math.Vector2()
        
        mr, mc = int(self.direction[0]//TILE_SIZE_x), int(self.direction[1]//TILE_SIZE_y)
        if self.wall_arr[mr][mc] == 1:
            self.direction = dir_orig


    def move(self, dt):
        # We'll only move if the direction of movement is not (0, 0), otherwise the 
        # Player will start moving to the left corner of the screen
    
        if self.direction.magnitude() != 0:
            # Move_towards function is only available in pygame 2.1.3dev8 and later
            # You can perharps do the same thing with pos.lerp(), but I've not tested it
            self.pos = self.pos.move_towards(self.direction, self.speed * dt)
            
        # Will only stop moving once the position reaches the destination
        if self.pos == self.direction:
            self.moving = False
        
        # You need to point the rect to the position, otherwise it will not appear to move at all
        self.rect.center = self.pos

    def update(self, dt):
        self.get_input()        
        self.move(dt)


class World:
    """
    The World class takes care of our World information.

    It contains our player and the current world.
    """
    def __init__(self):

        (prow, pcol) = 14,17 #12,5 #1,1 # initial position of PacMan (row, col)
        pacman_pos = (TILE_SIZE_x*prow,TILE_SIZE_y*pcol)
        
        g1row, g1col = 5,10 #1,8 # 16,5   # initial ghost1 position (row, col)
        g1_pos = (TILE_SIZE_x*g1row,TILE_SIZE_y*g1col)

        g2row, g2col = 3,10 #5,19  # 18,18  # initial ghost2 position (row, col)
        g2_pos = (TILE_SIZE_x*g2row,TILE_SIZE_y*g2col)

        g3row, g3col = 7,10 # 16,1  # initial ghost3 position (row, col)
        g3_pos = (TILE_SIZE_x*g3row,TILE_SIZE_y*g3col)
        
        g4row, g4col = 16,14  # initial ghost3 position (row, col)
        g4_pos = (TILE_SIZE_x*g4row,TILE_SIZE_y*g4col)

        wall_arr = self.processWalls()

        self.player = pygame.sprite.GroupSingle()
        Player(pacman_pos, wall_arr, self.player)
        
        self.initialize_nodes()

        self.g1 = pygame.sprite.GroupSingle()
        self.ghost1 = Ghost(g1_pos, 'red', self.g1)
        

        self.g2 = pygame.sprite.GroupSingle()
        self.ghost2 = Ghost(g2_pos, 'magenta', self.g2)
        
        
        self.g3 = pygame.sprite.GroupSingle()
        self.ghost3 = Ghost(g3_pos, 'orange', self.g3)

        self.g4 = pygame.sprite.GroupSingle()
        self.ghost4 = Ghost(g4_pos, 'cyan', self.g4)


        self.t = 0 # elapsed time 

    def visualize_nodeGrid(self, goal=None, start1=None, start2=None, start3=None, start4=None):
        ''' plot grid of nodes for both ghosts '''
        arr = np.zeros(self.node_grid.shape)
        
        for i in range(self.node_grid.shape[0]):
            for j in range(self.node_grid.shape[1]):
                if self.node_grid[j][i] == 0:
                    arr[i][j] = 1
        plt.imshow(arr)
        if goal is not None:
            plt.scatter(goal.row, goal.col, label='goal')
        if start1 is not None:
            plt.scatter(start1.row, start1.col, label='start1', c='red')
        if start2 is not None:
            plt.scatter(start2.row, start2.col, label='start2', c='magenta')
        if start3 is not None:
            plt.scatter(start3.row, start3.col, label='start3', c='orange')
        if start4 is not None:
            plt.scatter(start4.row, start4.col, label='start4', c='cyan')
        plt.legend()
        plt.show()

    def processWalls(self):
        ''' Process grid walls'''

        arr = np.zeros((ROWS, COLS))

        for row in range(ROWS):
            for col in range(COLS):
                if GRID[col][row] == '#':
                    arr[row][col] = 1

        return arr

    def initialize_nodes(self):
        ''' Initialize the grid of nodes for ghost to travel'''
        self.node_grid = np.zeros((ROWS, COLS), dtype='object')   ## node grid for ghost 
        for row in range(ROWS):
            for col in range(COLS):
                # Create a node per space, except only color walls black.
                if GRID[col][row] == '#':
                    self.node_grid[row][col] = 0
                else:
                    self.node_grid[row][col] = Node(row, col)
                    #nodes.append(Node(row, col))

        # Create the neighbors, being the edges between the nodes.
        flattened_nodes = self.node_grid.flatten()
        flattened_nodes = flattened_nodes[np.where(flattened_nodes != 0)]
        for node in flattened_nodes:
   
            for (dr, dc) in [(-1,0), (1,0), (0,-1), (0,1)]:

                others = [n for n in flattened_nodes 
                        if (n.row,n.col) == (node.row+dr,node.col+dc)]
                if len(others) > 0:
                    node.neighbors.append(others[0])


    def exclude_path_nodes(self, path):
        ''' reinitialize the grid of nodes but exclude the path nodes from ghost for the others'''

        #self.initialize_nodes()
        self.node_grid = np.zeros((ROWS, COLS), dtype='object')   ## node grid for ghost 
        for row in range(ROWS):
            for col in range(COLS):
                # Create a node per space, except only color walls black.
                if GRID[col][row] == '#':
                    self.node_grid[row][col] = 0
                else:
                    self.node_grid[row][col] = Node(row, col)

        for n in path:
            self.node_grid[n.row][n.col] = 0
        #self.visualize_nodeGrid()

        flattened_nodes = self.node_grid.flatten()
        flattened_nodes = flattened_nodes[np.where(flattened_nodes != 0)]
        for node in flattened_nodes:
   
            for (dr, dc) in [(-1,0), (1,0), (0,-1), (0,1)]:

                others = [n for n in flattened_nodes 
                        if (n.row,n.col) == (node.row+dr,node.col+dc)]
                if len(others) > 0:
                    node.neighbors.append(others[0])
        # self.visualize_nodeGrid()
                    
    def view_path(self, paths):
        """ Helper function to view paths of each ghost to goal
        """

        grid = np.zeros((21,21))
        sizes = [25, 15, 5]
        colors = ['red', 'magenta', 'orange']
        for i in range(21):
            for j in range(21):
                if GRID[i][j] == '#':
                    grid[i][j] = 1
        plt.imshow(grid*-1, cmap='gray')
        for i in range(len(paths)):
            for j in range(len(paths[i])):
                plt.scatter(paths[i][j].row,paths[i][j].col,color=colors[i],s=sizes[i] )
        plt.scatter(12,5,color='yellow',edgecolors='k')
        plt.show()


    def updateGhosts(self, pacPos, dt):
        # The player does not respond to input until it reaches the tile destination
        # That is the key to the Grid-Based movement here. If it could respond, then it would walk all over the place
        # if not self.moving:
        #     self.get_input()
       

        if (len(self.ghost1.path) == 0 and len(self.ghost2.path) == 0 and len(self.ghost3.path) == 0) or np.abs(self.t % 1 - 0) < 1e-2:

            # grab current positions of ghosts 
            g1_pos = self.ghost1.rect.topleft 
            g2_pos = self.ghost2.rect.topleft
            g3_pos = self.ghost3.rect.topleft

           # Prioritize G1
            self.initialize_nodes()
            self.ghost1.start_node = self.node_grid[int(g1_pos[0]//TILE_SIZE_x)][int(g1_pos[1]//TILE_SIZE_y)]
            self.ghost1.goal_node = self.node_grid[int(pacPos[0]//TILE_SIZE_x)][int(pacPos[1]//TILE_SIZE_y)]
            self.ghost1.get_path()


            # Prioritize G2
            self.initialize_nodes()
            self.ghost2.start_node = self.node_grid[int(g2_pos[0]//TILE_SIZE_x)][int(g2_pos[1]//TILE_SIZE_y)]
            self.ghost2.goal_node = self.node_grid[int(pacPos[0]//TILE_SIZE_x)][int(pacPos[1]//TILE_SIZE_y)]
            self.ghost2.get_path()

            # Prioritize G3
            self.initialize_nodes()
            self.ghost3.start_node = self.node_grid[int(g3_pos[0]//TILE_SIZE_x)][int(g3_pos[1]//TILE_SIZE_y)]
            self.ghost3.goal_node = self.node_grid[int(pacPos[0]//TILE_SIZE_x)][int(pacPos[1]//TILE_SIZE_y)]
            self.ghost3.get_path()

            # Uncomment to view each ghost path per time step
            #self.view_path([self.ghost1.path, self.ghost2.path, self.ghost3.path])
            

        for ghost in [self.ghost1, self.ghost2, self.ghost3]:
            if len(ghost.path) >= 1:
                while (len(ghost.path) > 0 and ghost.rect.topleft == ghost.path[-1].get_position()):
                    ghost.path.pop()
                if len(ghost.path) >= 1:   
                    ghost.move(ghost.path[-1].get_position(), dt)
              
        self.t = self.t + dt


            
    def update(self, dt):
        display = pygame.display.get_surface()
        self.player.update(dt)
        self.player.draw(display)

        pacPos = self.player.sprite.rect.topleft
        self.updateGhosts(pacPos, dt)
        self.g1.draw(display)
        self.g2.draw(display)
        self.g3.draw(display)


# We don't need a class, but it helps organize our code better
class Game:
    """
    Initializes pygame and handles events.
    """
    def __init__(self):
        pygame.init()
        # Initialized window and set SCALED for large resolution monitors
        self.window = pygame.display.set_mode([WINDOW_SIZE, WINDOW_SIZE]) #, pygame.SCALED)
        # Give a title to the window
        pygame.display.set_caption("Grid-Based Movement in Pygame")
        # You need the clock to get deltatime (dt)
        self.clock = pygame.time.Clock()
        self.world = World()
        # Control whether the program is running
        self.running = True
        # Show the Tile grid
        self.show_grid = True


    @staticmethod
    def draw_grid():
        """
        Draws the grid of tiles. Helps to visualize the grid-based movement.
        :return:
        """

        display = pygame.display.get_surface()
        gap_x = WINDOW_SIZE // COLS
        gap_y = WINDOW_SIZE // ROWS

        for i in range(ROWS):
            for j in range(COLS):

                if GRID[i][j] == '#':  
                    pygame.draw.rect(display, 'blue', (j*TILE_SIZE_y, i*TILE_SIZE_x, TILE_SIZE_y, TILE_SIZE_x ))

        for i in range(ROWS):
            pygame.draw.line(display, "white", (0, i * gap_y), (WINDOW_SIZE, i * gap_y))
            for j in range(COLS):
                pygame.draw.line(display, "white", (j * gap_x, 0), (j * gap_x, WINDOW_SIZE))

    def update(self, dt):
        self.window.fill("black")
        ret = self.world.update(dt)
        if self.show_grid:
            self.draw_grid()
        

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.K_d:  # toggles the grid
                    self.show_grid = not self.show_grid

            # Get deltatime for framerate independence
            self.clock.tick(7)
            dt = 10 #self.clock.tick() / 1000
            # Update game logic
            self.update(dt)
        
            # Similar to display.flip()
            pygame.display.update()

        pygame.quit()
        sys.exit()  # helps quit out of IDLE


if __name__ == "__main__":
    game = Game()
    game.run()
