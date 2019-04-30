import random
import math
import colorama

NORTH = (0, 1, 0)
EAST = (1, 0, 0)
UP = (0, 0, 1)
SOUTH = (0, -1, 0)
WEST = (-1, 0, 0)
DOWN = (0, 0, -1)
DIRS = [NORTH, EAST, UP, SOUTH, WEST, DOWN]


class CompatibilityOracle(object):

    """The CompatibilityOracle class is responsible for telling us
    which combinations of tiles and directions are compatible. It's
    so simple that it perhaps doesn't need to be a class, but I think
    it helps keep things clear.
    """

    def __init__(self, data):
        self.data = data

    def check(self, tile1, tile2, direction):
        return (tile1, tile2, direction) in self.data


class Wavefunction(object):

    """The Wavefunction class is responsible for storing which tiles
    are permitted and forbidden in each location of an output image.
    """

    @staticmethod
    def mk(size, weights):
        """Initialize a new Wavefunction for a grid of `size`,
        where the different tiles have overall weights `weights`.

        Arguments:
        size -- a 2-tuple of (width, height)
        weights -- a dict of tile -> weight of tile
        """
        coefficients = Wavefunction.init_coefficients(size, weights.keys())
        return Wavefunction(coefficients, weights)

    @staticmethod
    def init_coefficients(size, tiles):
        """Initializes a 3-D wavefunction grid of coefficients.
        The grid has size `size`, and each element of the grid
        starts with all tiles as possible. No tile is forbidden yet.

        NOTE: coefficients is a slight misnomer, since they are a
        set of possible tiles instead of a tile -> number/bool dict. This
        makes the code a little simpler. We keep the name `coefficients`
        for consistency with other descriptions of Wavefunction Collapse.

        Arguments:
        size -- a 2-tuple of (width, height, depth)
        tiles -- a set of all the possible tiles

        Returns:
        A 3-D grid in which each element is a set
        """
        coefficients = []

        for _ in range(size[0]):
            matrix = []
            for _ in range(size[1]):
                row = []
                for _ in range(size[2]):
                    row.append(set(tiles))
                matrix.append(row)
            coefficients.append(matrix)

        return coefficients

    def __init__(self, coefficients, weights):
        self.coefficients = coefficients
        self.weights = weights

    def get(self, co_ords):
        """Returns the set of possible tiles at `co_ords`"""
        x, y, z = co_ords
        return self.coefficients[x][y][z]

    def get_collapsed(self, co_ords):
        """Returns the only remaining possible tile at `co_ords`.
        If there is not exactly 1 remaining possible tile then
        this method raises an exception.
        """
        opts = self.get(co_ords)
        assert(len(opts) == 1)
        return next(iter(opts))

    def get_all_collapsed(self):
        """Returns a 3-D grid of the only remaining possible
        tiles at each location in the wavefunction. If any location
        does not have exactly 1 remaining possible tile then
        this method raises an exception.
        """
        width = len(self.coefficients)
        height = len(self.coefficients[0])
        depth = len(self.coefficients[0][0])

        collapsed = []

        for z in range(depth):
            matrix = []
            for x in range(width):
                row = []
                for y in range(height):
                    row.append(self.get_collapsed((x, y, z)))
                matrix.append(row)
            collapsed.append(matrix)

        return collapsed

    def shannon_entropy(self, co_ords):
        """Calculates the Shannon Entropy of the wavefunction at
        `co_ords`.
        """
        x, y, z = co_ords

        sum_of_weights = 0
        sum_of_weight_log_weights = 0
        for opt in self.coefficients[x][y][z]:
            weight = self.weights[opt]
            sum_of_weights += weight
            sum_of_weight_log_weights += weight * math.log(weight)

        return math.log(sum_of_weights) - (sum_of_weight_log_weights / sum_of_weights)

    def is_fully_collapsed(self):
        """Returns true if every element in Wavefunction is fully
        collapsed, and false otherwise.
        """
        for _, matrix in enumerate(self.coefficients):
            for _, row in enumerate(matrix):
                for _, sq in enumerate(row):
                    if len(sq) > 1:
                        return False

        return True

    def collapse(self, co_ords):
        """Collapses the wavefunction at `co_ords` to a single, definite
        tile. The tile is chosen randomly from the remaining possible tiles
        at `co_ords`, weighted according to the Wavefunction's global
        `weights`.

        This method mutates the Wavefunction, and does not return anything.
        """
        x, y, z = co_ords
        opts = self.coefficients[x][y][z]
        valid_weights = {tile: weight for tile,
                         weight in self.weights.iteritems() if tile in opts}

        total_weights = sum(valid_weights.values())
        rnd = random.random() * total_weights

        chosen = None
        for tile, weight in valid_weights.iteritems():
            rnd -= weight
            if rnd < 0:
                chosen = tile
                break

        self.coefficients[x][y][z] = set(chosen)

    def constrain(self, co_ords, forbidden_tile):
        """Removes `forbidden_tile` from the list of possible tiles
        at `co_ords`.

        This method mutates the Wavefunction, and does not return anything.
        """
        x, y, z = co_ords
        self.coefficients[x][y][z].remove(forbidden_tile)


class Model(object):

    """The Model class is responsible for orchestrating the
    Wavefunction Collapse algorithm.
    """

    def __init__(self, output_size, weights, compatibility_oracle):
        self.output_size = output_size
        self.compatibility_oracle = compatibility_oracle

        self.wavefunction = Wavefunction.mk(output_size, weights)

    def run(self):
        """Collapses the Wavefunction until it is fully collapsed,
        then returns a 3-D grid of the final, collapsed state.
        """
        while not self.wavefunction.is_fully_collapsed():
            self.iterate()

        return self.wavefunction.get_all_collapsed()

    def iterate(self):
        """Performs a single iteration of the Wavefunction Collapse
        Algorithm.
        """
        # 1. Find the co-ordinates of minimum entropy
        co_ords = self.min_entropy_co_ords()
        # 2. Collapse the wavefunction at these co-ordinates
        self.wavefunction.collapse(co_ords)
        # 3. Propagate the consequences of this collapse
        self.propagate(co_ords)

    def propagate(self, co_ords):
        """Propagates the consequences of the wavefunction at `co_ords`
        collapsing. If the wavefunction at (x,y) collapses to a fixed tile,
        then some tiles may not longer be theoretically possible at
        surrounding locations.

        This method keeps propagating the consequences of the consequences,
        and so on until no consequences remain.
        """
        stack = [co_ords]

        while len(stack) > 0:
            cur_coords = stack.pop()
            # Get the set of all possible tiles at the current location
            cur_possible_tiles = self.wavefunction.get(cur_coords)

            # Iterate through each location immediately adjacent to the
            # current location.
            for d in valid_dirs(cur_coords, self.output_size):
                other_coords = (
                    cur_coords[0] + d[0],
                    cur_coords[1] + d[1],
                    cur_coords[2] + d[2]
                )

                # Iterate through each possible tile in the adjacent location's
                # wavefunction.
                for other_tile in set(self.wavefunction.get(other_coords)):

                    # Check whether the tile is compatible with any tile in
                    # the current location's wavefunction.
                    other_tile_is_possible = any([
                        self.compatibility_oracle.check(
                            cur_tile, other_tile, d)
                        for cur_tile in cur_possible_tiles
                    ])

                    # If the tile is not compatible with any of the tiles in
                    # the current location's wavefunction then it is impossible
                    # for it to ever get chosen. We therefore remove it from
                    # the other location's wavefunction.
                    if not other_tile_is_possible:
                        self.wavefunction.constrain(other_coords, other_tile)
                        stack.append(other_coords)

    def min_entropy_co_ords(self):
        """Returns the co-ords of the location whose wavefunction has
        the lowest entropy.
        """
        min_entropy = None
        min_entropy_coords = None

        width, height, depth = self.output_size

        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if len(self.wavefunction.get((x, y, z))) == 1:
                        continue

                    entropy = self.wavefunction.shannon_entropy((x, y, z))

                    # Add some noise to mix things up a little
                    entropy_plus_noise = entropy - (random.random() / 1000)
                    if min_entropy is None or entropy_plus_noise < min_entropy:
                        min_entropy = entropy_plus_noise
                        min_entropy_coords = (x, y, z)

        return min_entropy_coords


def render_colors(grid, colors):
    """Render the fully collapsed `grid` using the given `colors`.

    Arguments:
    matrix -- 3-D grid of tiles
    colors -- dict of tile -> `colorama` color
    """
    print ""
    for matrix in grid:
        for row in matrix:
            output_row = []
            for val in row:
                color = colors[val]
                output_row.append(color + val + colorama.Style.RESET_ALL)

            print "".join(output_row)
        print ""


def valid_dirs(cur_co_ord, grid_size):
    """Returns the valid directions from `cur_co_ord` in a grid
    of `grid_size`. Ensures that we don't try to take step to the
    left when we are already on the left edge of the grid.
    """
    x, y, z = cur_co_ord
    width, height, depth = grid_size
    dirs = []

    if x > 0:
        dirs.append(WEST)
    if x < width-1:
        dirs.append(EAST)
    if y > 0:
        dirs.append(SOUTH)
    if y < height-1:
        dirs.append(NORTH)
    if z > 0:
        dirs.append(DOWN)
    if z < depth-1:
        dirs.append(UP)

    return dirs


def parse_example_grid(grid):
    """Parses an example `grid`. Extracts:

    1. Tile compatibilities - which pairs of tiles can be placed next
        to each other and in which directions
    2. Tile weights - how common different tiles are

    Arguments:
    grid -- a 3-D grid of matrixes

    Returns:
    A tuple of:
    * A set of compatibile tile combinations, where each combination is of
        the form (tile1, tile2, direction)
    * A dict of weights of the form tile -> weight
    """
    compatibilities = set()
    grid_depth = len(grid)
    grid_width = len(grid[0])
    grid_height = len(grid[0][0])

    weights = {}

    for z, matrix in enumerate(grid):
        for x, row in enumerate(matrix):
            for y, cur_tile in enumerate(row):
                if cur_tile not in weights:
                    weights[cur_tile] = 0
                weights[cur_tile] += 1

                print('sizes', grid_width, grid_height, grid_depth)
                for d in valid_dirs((x, y, z), (grid_width, grid_height, grid_depth)):
                    other_tile = grid[z+d[2]][x+d[0]][y+d[1]]
                    compatibilities.add((cur_tile, other_tile, d))

    return compatibilities, weights


input_grid = [
    [
        ['C', 'C', 'C', 'C'],
        ['C', 'A', 'A', 'C'],
        ['C', 'C', 'C', 'C'],
    ],
    [
        ['S', 'S', 'S', 'S'],
        ['S', 'A', 'A', 'S'],
        ['S', 'S', 'S', 'S'],
    ],
    [
        ['L', 'L', 'L', 'L'],
        ['L', 'A', 'A', 'L'],
        ['L', 'L', 'L', 'L'],
    ],
]

compatibilities, weights = parse_example_grid(input_grid)
compatibility_oracle = CompatibilityOracle(compatibilities)
model = Model((10, 20, 3), weights, compatibility_oracle)
output = model.run()

colors = {
    'L': colorama.Fore.GREEN,
    'S': colorama.Fore.BLUE,
    'C': colorama.Fore.YELLOW,
    'A': colorama.Fore.CYAN,
    'B': colorama.Fore.MAGENTA,
}

render_colors(output, colors)
