import numpy as np
import matplotlib.pyplot as plt

import itbl._itbl as _itbl


def uniform_sample_maze(maze_dims, hall_width, wall_width):
    # Create set of all possible walls.
    walls = set()
    for i in range(maze_dims[0]):
        for j in range(maze_dims[1]):
            p = (i, j)
            q = (i, j - 1)
            walls.add((min(p, q), max(p, q)))
            q = (i - 1, j)
            walls.add((min(p, q), max(p, q)))
            q = (i, j + 1)
            walls.add((min(p, q), max(p, q)))
            q = (i + 1, j)
            walls.add((min(p, q), max(p, q)))

    # Set of tiles to added to the maze.
    maze = set()
    maze.add((0, 0))
    tiles = set()
    for i in range(maze_dims[0]):
        for j in range(maze_dims[1]):
            if not (i, j) in maze:
                tiles.add((i, j))

    # Do loop-erased random walk until all tiles are added to the maze.
    while tiles:
        # Sample random tile.
        r = np.random.choice(len(tiles))
        tile = list(tiles)[r]

        # Random walk until we hit a tile in the maze.
        stack = np.prod(maze_dims) * [None]
        offset = 0
        stack[offset] = tile
        while True:
            # Sample next tile.
            t = stack[offset]
            d = np.random.randint(0, 4)
            if d == 0:
                n = (t[0] + 1, t[1])
            if d == 1:
                n = (t[0], t[1] + 1)
            if d == 2:
                n = (t[0] - 1, t[1])
            if d == 3:
                n = (t[0], t[1] - 1)

            # Reject if out of bounds.
            if (n[0] < 0) or (maze_dims[0] <= n[0]):
                continue
            if (n[1] < 0) or (maze_dims[1] <= n[1]):
                continue

            # Add tile to stack.
            offset += 1
            stack[offset] = n

            if n in maze:
                # Add walk to maze.
                for i in range(offset):
                    maze.add(stack[i])
                    tiles.remove(stack[i])
                    p = stack[i]
                    q = stack[i + 1]
                    walls.remove((min(p, q), max(p, q)))
                break
            else:
                # Erase loops.
                for i in range(offset):
                    if n == stack[i]:
                        offset = i
                        break

    # Construct maze.
    manager = _itbl.CollisionManager2D()
    for e in walls:
        v0 = np.array(e[0])
        v1 = np.array(e[1])

        wall = _itbl.Rectangle(hall_width + 2 * wall_width, wall_width, 2, 0.05)

        wall.transform()[0:2, 3] = (hall_width + wall_width) * (v0 + v1) / 2.0
        wall.transform()[2, 3] = 0
        dy = v0[1] - v1[1]
        if dy == 0:
            wall.transform()[0:2, 0:2] = np.array([[0, -1], [1, 0]])

        manager.add(wall)

    return manager


def create_hallway(hall_width, block_width, block_height, center_x):
    # Construct wall.
    manager = _itbl.CollisionManager2D()
    wall1 = _itbl.Rectangle(block_width, block_height, 2, 0.05)
    wall2 = _itbl.Rectangle(block_width, block_height, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([center_x + block_width / 2, block_height / 2, 0]).reshape(
        wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array(
        [center_x + block_width / 2, block_height / 2 + block_height + hall_width, 0]).reshape(
        wall1.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)

    return manager


def corner():
    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(5, 1, 2, 0.05)
    wall2 = _itbl.Rectangle(1, 4, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([1.5, -0.5, 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([-0.5, 2, 0]).reshape(wall2.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)

    return manager

def table():
    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(5, 1, 2, 0.05)
    wall1.transform()[0:3, 3] = np.array([1.5, -0.5, 0]).reshape(wall1.transform()[0:3, 3].shape)

    manager.add(wall1)

    return manager


def wall():
    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(8, 3, 2, 0.05)
    wall2 = _itbl.Rectangle(8, 10, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([4, -1.5, 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([-4, 2, 0]).reshape(wall2.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)

    return manager


def table():
    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(20, 3, 2, 0.05)
    wall1.transform()[0:3, 3] = np.array([0, -1.5, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall1)

    return manager


def obstacle_course():
    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(8, 2, 2, 0.05)
    wall2 = _itbl.Rectangle(1, 0.5, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([0, 0, 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([0, 1.25, 0]).reshape(wall2.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)

    return manager

def peg_in_hole_v():

    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(3, 4, 2, 0.05)
    wall2 = _itbl.Rectangle(3, 4, 2, 0.05)
    wall0 = _itbl.Rectangle(1, 2, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([-2, -2 , 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([2, -2, 0]).reshape(wall2.transform()[0:3, 3].shape)
    wall0.transform()[0:3, 3] = np.array([0, -3, 0]).reshape(wall2.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)
    manager.add(wall0)

    return manager

def peg_in_hole_p():

    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(5, 3, 2, 0.05)
    wall2 = _itbl.Rectangle(3, 1, 2, 0.05)
    wall0 = _itbl.Rectangle(10, 3, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([-2.5, 2.5 , 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([-3.5, 0.5, 0]).reshape(wall2.transform()[0:3, 3].shape)
    wall0.transform()[0:3, 3] = np.array([0, -1.5, 0]).reshape(wall2.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)
    manager.add(wall0)

    return manager

def book():
    manager = _itbl.CollisionManager2D()
    wall1 = _itbl.Rectangle(6, 4, 2, 0.05)
    wall1.transform()[0:3, 3] = np.array([0, 0, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall1)

    return manager

def unpacking():

    manager = _itbl.CollisionManager2D()

    wall1 = _itbl.Rectangle(4, 1, 2, 0.05)
    wall2 = _itbl.Rectangle(1, 2, 2, 0.05)
    wall0 = _itbl.Rectangle(2, 2, 2, 0.05)

    wall1.transform()[0:3, 3] = np.array([0, -1.5 , 0]).reshape(wall1.transform()[0:3, 3].shape)
    wall2.transform()[0:3, 3] = np.array([-1.5, 0, 0]).reshape(wall2.transform()[0:3, 3].shape)
    wall0.transform()[0:3, 3] = np.array([1, 0, 0]).reshape(wall0.transform()[0:3, 3].shape)

    manager.add(wall1)
    manager.add(wall2)
    manager.add(wall0)

    return manager

def pushing():
    manager = _itbl.CollisionManager2D()
    wall1 = _itbl.Rectangle(9, 1.5, 2, 0.05)
    wall1.transform()[0:3, 3] = np.array([0, -3.25, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall1)
    wall2 = _itbl.Rectangle(1.5, 6.5, 2, 0.05)
    wall2.transform()[0:3, 3] = np.array([-3.75, 0.75, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall2)
    wall3 = _itbl.Rectangle(4, 2.5, 2, 0.05)
    wall3.transform()[0:3, 3] = np.array([1, -1.25, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall3)
    wall4 = _itbl.Rectangle(1.5, 6.5, 2, 0.05)
    wall4.transform()[0:3, 3] = np.array([3.75, 0.75, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall4)
    wall5 = _itbl.Rectangle(2.4, 2.7, 2, 0.05)
    wall5.transform()[0:3, 3] = np.array([0, 2.65, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall5)
    wall6 = _itbl.Rectangle(9, 1.5, 2, 0.05)
    wall6.transform()[0:3, 3] = np.array([0, 4.75, 0]).reshape(wall1.transform()[0:3, 3].shape)
    manager.add(wall6)

    return manager