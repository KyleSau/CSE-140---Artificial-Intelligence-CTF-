"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging
from pacai.core.actions import Actions
from pacai.core.search import heuristic
from pacai.core.directions import Directions
from pacai.core.distance import maze
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.core.search.search import uniformCostSearch
from pacai.student import search


class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    `pacai.core.search.problem.SearchProblem.isGoal`:
    Returns whether this search state is a goal state of the problem.

    `pacai.core.search.problem.SearchProblem.successorStates`:
    Returns successor states, the actions they require, and a cost of 1.
    The following code snippet may prove useful:
    ```
        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                # Construct the successor.

        return successors
    ```
    """

    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

    def startingState(self):
        return (self.startingPosition, tuple([corner for corner in self.corners]))

    def isGoal(self, state):
        # Goal check for corners
        state, visited_corners = state
        # Register the locations we have visited.
        # This allows the GUI to highlight them.
        self._visitedLocations.add(state)
        # Note: visit history requires coordinates not states. In this situation
        # they are equivalent.
        coordinates = state
        self._visitHistory.append(coordinates)
        return True if len(list(visited_corners)) == 0 else False

    def successorStates(self, state):
        state, visited_corners = state
        successors = []

        for action in Directions.CARDINAL:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)

            if (not self.walls[nextx][nexty]):
                nextState = (nextx, nexty)
                cost = 1
                if nextState in visited_corners:
                    visited_corner_e = list(visited_corners)
                    del visited_corner_e[visited_corner_e.index(nextState)]
                    successors.append(((nextState, tuple(visited_corner_e)), action, cost))
                else:
                    successors.append(((nextState, visited_corners), action, cost))
        # Bookkeeping for display purposes (the highlight in the GUI).
        self._numExpanded += 1
        if (state not in self._visitedLocations):
            self._visitedLocations.add(state)
            # Note: visit history requires coordinates not states. In this situation
            # they are equivalent.
            coordinates = state
            self._visitHistory.append(coordinates)

        return successors

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """

        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***
    state, corners = state
    x_curr, y_curr = state
    if len(corners) == 0:
        return heuristic.null(state, problem)
    l_corners = list(corners)
    total_cost = 0
    while len(l_corners) > 0:
        d_table = {i: 999999 for i in l_corners}
        for corner in l_corners:
            x_goal, y_goal = corner
            dx, dy = abs(x_goal - x_curr), abs(y_goal - y_curr)
            d_table[corner] = dx + dy
        x_curr, y_curr = x_goal, y_goal
        nextState = min(d_table, key=d_table.get)
        x_goal, y_goal = nextState
        total_cost += d_table[nextState]
        del l_corners[l_corners.index(nextState)]
    return total_cost
    # return heuristic.null(state, problem)  # Default to trivial solution

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """
    position, foodGrid = state
    # *** Your Code Here ***
    foods = foodGrid.asList()
    if len(foods) < 1:
        return heuristic.null(state, problem)
    d_table = {}
    for food in foods:
        d_table[food] = maze(position, food, problem.startingGameState)
    sorted_food = list(sorted(d_table.items(), key=lambda item: item[1]))
    pos_max, d_max = sorted_food[-1]
    if len(sorted_food) < 2:
        return d_max
    pos_max2, d_max2 = sorted_food[-2]
    d_final = maze(pos_max, pos_max2, problem.startingGameState)
    x_curr, y_curr = position
    for action in list(maze_act(position, pos_max,
            problem.startingGameState) + maze_act(pos_max, pos_max2, problem.startingGameState)):
        if (x_curr, y_curr) in foods:
            del foods[foods.index((x_curr, y_curr))]
        if action == "North":
            y_curr += 1
        elif action == "South":
            y_curr -= 1
        elif action == "West":
            x_curr -= 1
        else:
            x_curr += 1
    return (d_final + d_max2) / 2 + len(foods)

class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # *** Your Code Here ***
        problem = AnyFoodSearchProblem(gameState=gameState)
        return uniformCostSearch(problem)


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()
        foods = self.food.asList()
        dist_table = {i: maze(gameState.getPacmanPosition(), i, gameState) for i in foods}
        sorted_food = list(sorted(dist_table.items(), key=lambda item: item[1]))
        self.goal_pos, d_min = sorted_food[0]

    def isGoal(self, state):
        return True if state == self.goal_pos else False

class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

def maze_act(position1, position2, gameState):
    """
    Returns the maze distance between any two positions,
    using the search functions you have already built.

    WARNING: `pacai.student.search.breadthFirstSearch` must already be implemted.

    Example usage: `distance.maze((2, 4), (5, 6), gameState)`.
    """

    x1, y1 = position1
    x2, y2 = position2

    walls = gameState.getWalls()

    if (walls[x1][y1]):
        raise ValueError('Position1 is a wall: ' + str(position1))

    if (walls[x2][y2]):
        raise ValueError('Position2 is a wall: ' + str(position2))

    prob = PositionSearchProblem(gameState, start = position1, goal = position2)

    return search.breadthFirstSearch(prob)
