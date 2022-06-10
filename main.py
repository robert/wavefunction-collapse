import random
import math
from typing import Literal
import colorama

# Types
Tile = str
Coordinates = tuple[int, int]
Up = tuple[Literal[1], Literal[0]]; Down = tuple[Literal[-1], Literal[0]]
Left = tuple[Literal[0], Literal[-1]]; Right = tuple[Literal[0], Literal[1]]
Direction = Up | Down | Left | Right
Compatibility = tuple[Tile, Tile, Direction]
Weights = dict[Tile, int]
Coefficients = set[Tile]
CoefficientMatrix = list[list[Coefficients]]

UP = (1, 0)
DOWN = (-1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = [UP, DOWN, LEFT, RIGHT]


class CompatibilityOracle(object):

    """The CompatibilityOracle class is responsible for telling us
    which combinations of tiles and directions are compatible. It's
    so simple that it perhaps doesn't need to be a class, but I think
    it helps keep things clear.
    """

    def __init__(self, data: set[Compatibility]):
        self.data = data

    def check(self, tile1: Tile, tile2: Tile, direction: Direction) -> bool:
        return (tile1, tile2, direction) in self.data


class Wavefunction(object):

    """The Wavefunction class is responsible for storing which tiles
    are permitted and forbidden in each location of an output image.
    """

    @staticmethod
    def mk(size: tuple[int, int], weights: Weights):
        """Initialize a new Wavefunction for a grid of `size`,
        where the different tiles have overall weights `weights`.

        Arguments:
        size -- a 2-tuple of (width, height)
        weights -- a dict of tile -> weight of tile
        """
        coefficient_matrix = Wavefunction.init_coefficient_matrix(size, list(weights.keys()))
        return Wavefunction(coefficient_matrix, weights)

    @staticmethod
    def init_coefficient_matrix(size: tuple[int, int], tiles: list[Tile]) -> CoefficientMatrix:
        """Initializes a 2-D wavefunction matrix of coefficients.
        The matrix has size `size`, and each element of the matrix
        starts with all tiles as possible. No tile is forbidden yet.

        NOTE: coefficients is a slight misnomer, since they are a
        set of possible tiles instead of a tile -> number/bool dict. This
        makes the code a little simpler. We keep the name `coefficients`
        for consistency with other descriptions of Wavefunction Collapse.

        Arguments:
        size -- a 2-tuple of (width, height)
        tiles -- a set of all the possible tiles

        Returns:
        A 2-D matrix in which each element is a set
        """
        coefficient_matrix: CoefficientMatrix = []

        for _ in range(size[1]):
            row: list[Coefficients] = []
            for _ in range(size[0]):
                row.append(set(tiles))
            coefficient_matrix.append(row)

        return coefficient_matrix

    def __init__(self, coefficient_matrix: CoefficientMatrix, weights: Weights):
        self.coefficient_matrix = coefficient_matrix
        self.weights = weights

    def get(self, co_ords: Coordinates) -> Coefficients:
        """Fetches the set of possible tiles at `co_ords`.

        Arguments:
        co_ords -- Tuple representing 2D co-ordinates in the format (y, x).

        Returns:
        The set of possible tiles.
        """
        y, x = co_ords
        return self.coefficient_matrix[y][x]

    def get_collapsed(self, co_ords: Coordinates) -> Tile:
        """Returns the only remaining possible tile at `co_ords`.
        If there is not exactly 1 remaining possible tile then
        this method raises an exception.
        """
        opts = self.get(co_ords)
        assert(len(opts) == 1)
        return next(iter(opts))

    def get_all_collapsed(self) -> list[list[Tile]]:
        """Returns a 2-D matrix of the only remaining possible
        tiles at each location in the wavefunction. If any location
        does not have exactly 1 remaining possible tile then
        this method raises an exception.
        """
        height = len(self.coefficient_matrix)
        width = len(self.coefficient_matrix[0])

        collapsed: list[list[Tile]] = []
        for y in range(height):
            row: list[Tile] = []
            for x in range(width):
                row.append(self.get_collapsed((y, x)))
            collapsed.append(row)

        return collapsed

    def shannon_entropy(self, co_ords: Coordinates) -> float:
        """Calculates the Shannon Entropy of the wavefunction at
        `co_ords`.
        """
        y, x = co_ords

        sum_of_weights = 0
        sum_of_weight_log_weights = 0
        for opt in self.coefficient_matrix[y][x]:
            weight = self.weights[opt]
            sum_of_weights += weight
            sum_of_weight_log_weights += weight * math.log(weight)

        return math.log(sum_of_weights) - (sum_of_weight_log_weights / sum_of_weights)


    def is_fully_collapsed(self) -> bool:
        """Returns true if every element in Wavefunction is fully
        collapsed, and false otherwise.
        """
        for row in self.coefficient_matrix:
            for sq in row:
                if len(sq) > 1:
                    return False

        return True

    def collapse(self, co_ords: Coordinates) -> None:
        """Collapses the wavefunction at `co_ords` to a single, definite
        tile. The tile is chosen randomly from the remaining possible tiles
        at `co_ords`, weighted according to the Wavefunction's global
        `weights`.

        This method mutates the Wavefunction, and does not return anything.
        """
        y, x = co_ords
        opts = self.coefficient_matrix[y][x]
        filtered_tiles_with_weights = [(tile, weight) for tile, weight in self.weights.items() if tile in opts]

        total_weights = sum([weight for _, weight in filtered_tiles_with_weights])
        rnd = random.random() * total_weights

        chosen = filtered_tiles_with_weights[0][0]
        for tile, weight in filtered_tiles_with_weights:
            rnd -= weight
            if rnd < 0:
                chosen = tile
                break

        self.coefficient_matrix[y][x] = set([chosen])

    def constrain(self, co_ords: Coordinates, forbidden_tile: Tile) -> None:
        """Removes `forbidden_tile` from the list of possible tiles
        at `co_ords`.

        This method mutates the Wavefunction, and does not return anything.
        """
        y, x = co_ords
        self.coefficient_matrix[y][x].remove(forbidden_tile)


class Model(object):

    """The Model class is responsible for orchestrating the
    Wavefunction Collapse algorithm.
    """

    def __init__(self, output_size: tuple[int, int], weights: Weights, compatibility_oracle: CompatibilityOracle):
        self.output_size = output_size
        self.compatibility_oracle = compatibility_oracle

        self.wavefunction = Wavefunction.mk(output_size, weights)

    def run(self) -> list[list[Tile]]:
        """Collapses the Wavefunction until it is fully collapsed,
        then returns a 2-D matrix of the final, collapsed state.
        """
        while not self.wavefunction.is_fully_collapsed():
            self.iterate()

        return self.wavefunction.get_all_collapsed()

    def iterate(self) -> None:
        """Performs a single iteration of the Wavefunction Collapse
        Algorithm.
        """
        # 1. Find the co-ordinates of minimum entropy
        co_ords = self.min_entropy_co_ords()
        # 2. Collapse the wavefunction at these co-ordinates
        self.wavefunction.collapse(co_ords)
        # 3. Propagate the consequences of this collapse
        self.propagate(co_ords)

    def propagate(self, co_ords: Coordinates) -> None:
        """Propagates the consequences of the wavefunction at `co_ords`
        collapsing. If the wavefunction at (y, x) collapses to a fixed tile,
        then some tiles may not longer be theoretically possible at
        surrounding locations.

        This method keeps propagating the consequences of the consequences,
        and so on until no consequences remain.
        """
        stack = [co_ords]

        while len(stack) > 0:
            cur_co_ords = stack.pop()
            # Get the set of all possible tiles at the current location
            cur_possible_tiles = self.wavefunction.get(cur_co_ords)

            # Iterate through each location immediately adjacent to the
            # current location.
            for d in valid_dirs(cur_co_ords, self.output_size):
                other_co_ords = (cur_co_ords[0] + d[0], cur_co_ords[1] + d[1])

                # Iterate through each possible tile in the adjacent location's
                # wavefunction.
                for other_tile in set(self.wavefunction.get(other_co_ords)):
                    # Check whether the tile is compatible with any tile in
                    # the current location's wavefunction.
                    other_tile_is_possible = any([
                        self.compatibility_oracle.check(cur_tile, other_tile, d) for cur_tile in cur_possible_tiles
                    ])
                    # If the tile is not compatible with any of the tiles in
                    # the current location's wavefunction then it is impossible
                    # for it to ever get chosen. We therefore remove it from
                    # the other location's wavefunction.
                    if not other_tile_is_possible:
                        self.wavefunction.constrain(other_co_ords, other_tile)
                        stack.append(other_co_ords)

    def min_entropy_co_ords(self) -> Coordinates:
        """Returns the co-ords of the location whose wavefunction has
        the lowest entropy.
        """
        min_entropy = None
        min_entropy_co_ords: Coordinates = (0, 0)

        width, height = self.output_size
        for y in range(height):
            for x in range(width):
                if len(self.wavefunction.get((y, x))) == 1:
                    continue

                entropy = self.wavefunction.shannon_entropy((y, x))
                # Add some noise to mix things up a little
                entropy_plus_noise = entropy - (random.random() / 1000)
                if min_entropy is None or entropy_plus_noise < min_entropy:
                    min_entropy = entropy_plus_noise
                    min_entropy_co_ords = (y, x)

        return min_entropy_co_ords


def render_colors(matrix: list[list[Tile]], colors: dict[str, str]) -> None:
    """Render the fully collapsed `matrix` using the given `colors.

    Arguments:
    matrix -- 2-D matrix of tiles
    colors -- dict of tile -> `colorama` color
    """
    for row in matrix:
        output_row: list[str] = []
        for val in row:
            color = colors[val]
            output_row.append(color + val + colorama.Style.RESET_ALL)

        print("".join(output_row))


def valid_dirs(cur_co_ords: Coordinates, matrix_size: tuple[int, int]) -> list[Direction]:
    """Returns the valid directions from `cur_co_ord` in a matrix
    of `matrix_size`. Ensures that we don't try to take step to the
    left when we are already on the left edge of the matrix.
    """
    y, x = cur_co_ords
    width, height = matrix_size
    dirs: list[Direction] = []

    if y < height-1: dirs.append(UP)
    if y > 0: dirs.append(DOWN)
    if x > 0: dirs.append(LEFT)
    if x < width-1: dirs.append(RIGHT)

    return dirs


def parse_example_matrix(matrix: list[list[Tile]]) -> tuple[set[Compatibility], Weights]:
    """Parses an example `matrix`. Extracts:
    
    1. Tile compatibilities - which pairs of tiles can be placed next
        to each other and in which directions
    2. Tile weights - how common different tiles are

    Arguments:
    matrix -- a 2-D matrix of tiles

    Returns:
    A tuple of:
    * A set of compatibile tile combinations, where each combination is of
        the form (tile1, tile2, direction)
    * A dict of weights of the form tile -> weight
    """
    compatibilities: set[Compatibility] = set()
    matrix_height = len(matrix)
    matrix_width = len(matrix[0])

    weights: Weights = {}

    for y, row in enumerate(matrix):
        for x, cur_tile in enumerate(row):
            if cur_tile not in weights:
                weights[cur_tile] = 0
            weights[cur_tile] += 1

            for d in valid_dirs((y, x), (matrix_width, matrix_height)):
                other_tile = matrix[y+d[0]][x+d[1]]
                compatibilities.add((cur_tile, other_tile, d))

    return compatibilities, weights


input_matrix = [
    ['L','L','L','L'],
    ['L','L','L','L'],
    ['L','L','L','L'],
    ['L','C','C','L'],
    ['C','S','S','C'],
    ['S','S','S','S'],
    ['S','S','S','S'],
]
input_matrix2 = [
    ['A','A','A','A'],
    ['A','A','A','A'],
    ['A','A','A','A'],
    ['A','C','C','A'],
    ['C','B','B','C'],
    ['C','B','B','C'],
    ['A','C','C','A'],
]

compatibilities, weights = parse_example_matrix(input_matrix)
compatibility_oracle = CompatibilityOracle(compatibilities)
model = Model((50, 10), weights, compatibility_oracle)
output = model.run()

colors = {
    'L': colorama.Fore.GREEN,
    'S': colorama.Fore.BLUE,
    'C': colorama.Fore.YELLOW,
    'A': colorama.Fore.CYAN,
    'B': colorama.Fore.MAGENTA,
}

render_colors(output, colors)
