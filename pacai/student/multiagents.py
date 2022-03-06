import random
from operator import itemgetter
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        nextGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***

        return betterEvaluationFunction(nextGameState) - betterEvaluationFunction(currentGameState)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def miniMax(state, depth, agent):
            legalActions = list(state.getLegalActions(agent))
            if 'Stop' in legalActions:
                legalActions.remove('Stop')
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state), 'Stop'
            action_value = {}
            # agent == 0 means its pacman
            if agent == 0:
                # max value function
                value = -999999
                for action in legalActions:
                    nextState = state.generateSuccessor(agent, action)
                    action_value[action], f_ac = miniMax(nextState, depth, agent + 1)
                    value = max([value, action_value[action]])
                return value, max(action_value.items(), key = itemgetter(1))[0]
            else:
                # min value function
                value = 999999
                for action in legalActions:
                    nextState = state.generateSuccessor(agent, action)
                    if agent < (state.getNumAgents() - 1):
                        # forgot to account for index
                        action_value[action], f_ac = miniMax(nextState, depth, agent + 1)
                        value = min([value, action_value[action]])
                    else:
                        # ran through all agents once, so now we need to increase depth
                        action_value[action], f_ac = miniMax(nextState, depth + 1, 0)
                        value = min([value, action_value[action]])
                return value, min(action_value.items(), key = itemgetter(1))[0]
        value, action = miniMax(gameState, 0, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def alphaBetaPruning(state, depth, agent, a, b):
            legalActions = list(state.getLegalActions(agent))
            if 'Stop' in legalActions:
                legalActions.remove('Stop')
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state), 'Stop'
            action_v = {}
            # agent == 0 means its pacman
            if agent == 0:
                # max value function
                value = -999999
                for action in legalActions:
                    nextState = state.generateSuccessor(agent, action)
                    action_v[action], f_ac = alphaBetaPruning(nextState, depth, agent + 1, a, b)
                    value = max([value, action_v[action]])
                    if value >= b:
                        return value, action
                    a = max([a, value])
                return value, max(action_v.items(), key = itemgetter(1))[0]
            else:
                # min value function
                value = 999999
                for action in legalActions:
                    nextState = state.generateSuccessor(agent, action)
                    if agent < (state.getNumAgents() - 1):
                        # forgot to account for index
                        action_v[action], f_ac = alphaBetaPruning(nextState, depth, agent + 1, a, b)
                        value = min([value, action_v[action]])
                    else:
                        # ran through all agents once, so now we need to increase depth
                        action_v[action], f_ac = alphaBetaPruning(nextState, depth + 1, 0, a, b)
                        value = min([value, action_v[action]])
                    if value <= a:
                        return value, action
                    b = min([b, value])
                return value, min(action_v.items(), key = itemgetter(1))[0]
        value, action = alphaBetaPruning(gameState, 0, 0, -999999, 999999)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        def expectiMax(state, depth, agent):
            legalActions = list(state.getLegalActions(agent))
            if 'Stop' in legalActions:
                legalActions.remove('Stop')
            if depth == self.getTreeDepth() or state.isWin() or state.isLose():
                return self.getEvaluationFunction()(state), 'Stop'
            action_v = {}
            # agent == 0 means its pacman
            if agent == 0:
                # max value function
                value = -999999
                for action in legalActions:
                    nextState = state.generateSuccessor(agent, action)
                    action_v[action], f_ac = expectiMax(nextState, depth, agent + 1)
                    value = max([value, action_v[action]])
                return value, max(action_v.items(), key = itemgetter(1))[0]
            else:
                # avg utility value function
                value = 999999
                for action in legalActions:
                    nextState = state.generateSuccessor(agent, action)
                    if agent < (state.getNumAgents() - 1):
                        # forgot to account for index
                        action_v[action], f_ac = expectiMax(nextState, depth, agent + 1)
                    else:
                        # ran through all agents once, so now we need to increase depth
                        action_v[action], f_ac = expectiMax(nextState, depth + 1, 0)
                    mean = sum(list(action_v.values())) / len(action_v.values())
                return mean, list(action_v.keys())[0]
        value, action = expectiMax(gameState, 0, 0)
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.
    DESCRIPTION:
    The final score returned should have the following correlations:
    - final score should be inversely coorelated with closest distance to scared ghost
    (lower d = lower score)
    - final score should be directly coorelated with closest distance to active ghost
    (high d = higher score)
    - final score should be inversely correlated with closest distance to food
    - The final score should never be zeroed out, which means the weights should be added.
    - The active ghost weight (which gets subtracted if theres an active ghost nearby) is
    multiplied by 2 in order to ensure we move away from an active ghost running at us
    - In order to ensure actions from the ghost always get higher priority than food placement,
    the weight for food is divided by 2
    """
    position = currentGameState.getAgentPosition(0)
    x_curr, y_curr = position
    curr_Score = currentGameState.getScore()
    d_food = {}  # distance to foods
    d_active_ghost = {}  # distance to active ghosts
    d_scared_ghost = {}  # distance to scared ghosts
    for food in currentGameState.getFood().asList():
        # figure out distance to each food pellet
        d_food[food] = manhattan(position, food)
    for ghost in currentGameState.getGhostStates():
        # get ghost distances
        ghost_x, ghost_y = ghost.getPosition()
        ghost_pos = (int(ghost_x), int(ghost_y))
        if ghost.isScaredGhost():
            d_scared_ghost[ghost_pos] = manhattan(position, ghost_pos)
        else:
            d_active_ghost[ghost_pos] = manhattan(position, ghost_pos)
    # The following section ensures weights are defaulted to 0 when no distances are found
    min_scared_ghost_d = 0
    min_active_ghost_d = 0
    min_food_d = 0
    if len(d_scared_ghost) > 0:
        min_scared_ghost_d = min(list(d_scared_ghost.values()))
    if len(d_active_ghost) > 0:
        min_active_ghost_d = min(list(d_active_ghost.values()))
    if len(d_food) > 0:
        min_food_d = min(list(d_food.values()))
    # The following section creates a weight for each variable using their minimum distances
    food_weight = 1 / min_food_d if min_food_d > 0 else 0
    active_ghost_weight = 1 / min_active_ghost_d if min_active_ghost_d > 0 else 0
    scared_ghost_weight = 1 / min_scared_ghost_d if min_scared_ghost_d > 0 else 0
    return curr_Score + food_weight / 2 - active_ghost_weight * 2 + scared_ghost_weight

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
