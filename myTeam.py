from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util.priorityQueue import PriorityQueue
from pacai.core.search.position import PositionSearchProblem
from pacai.core.distance import euclidean
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        OffensiveReflexAgent(firstIndex),
        DefensiveReflexAgent(secondIndex),
    ]
    
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
        self.startingState = start

    def getStartState(self):
        return self.startingState

    def isGoalState(self, state):
        x, y = state

        if self.food[x][y]:
            return True
        return False

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
        
    def getNearestGhostPosition(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        nonScaredGhosts = {a.getPosition(): self.getMazeDistance(gameState.getAgentPosition(self.index), a.getPosition()) for a in enemies if not a.isPacman()}
        return min(nonScaredGhosts, key=nonScaredGhosts.get)
    
    """
    def aStarSearch(self, problem, gameState, heuristic):
        start_state = problem.getStartState()
        fringe = PriorityQueue()
        print(start_state, gameState)
        h = heuristic(gameState)
        g = 0
        f = g + h
        start_node = (start_state, [], g)
        fringe.push(start_node, f)
        explored = []
        while not fringe.isEmpty():
          current_node = fringe.pop()
          state = current_node[0]
          path = current_node[1]
          current_cost = current_node[2]
          if state not in explored:
            explored.append(state)
            if problem.isGoalState(state):
              return path
            for successor in gameState.getLegalActions(self.index):
                # successor = self.getSuccessor(gameState, action)
                current_path = list(path)
                successor_state = successor[0]
                move = successor[1]
                g = successor[2] + current_cost
                h = heuristic(gameState)
                if successor_state not in explored:
                    current_path.append(move)
                    f = g + h
                    successor_node = (successor_state, current_path, g)
                    fringe.push(successor_node, f)
        return []
        """
       
    def aStarSearch(self, problem, gameState, heuristic):
        """
        Search the node that has the lowest combined cost and heuristic first.
        """

        # *** Your Code Here ***
        fringe = PriorityQueue()
        start_state = problem.getStartState()
        h = heuristic(start_state, gameState)
        g = 0
        f = g + h
        start_node = (start_state, [], g)
        fringe.push(start_node, f)
        visitedStates = set()
        while not fringe.isEmpty():
            state, action, cost = fringe.pop()
            if problem.isGoal(state):
                return action
            if state not in visitedStates:
                visitedStates.add(state)
                for child in problem.successorStates(state):
                    fringe.push([child[0], action + [child[1]], cost + child[2]], cost + heuristic(child[0], problem))
        return None
        
    def cheesyAStar(position1, position2, gameState, enemies):
        enemyGhosts = [a for a in enemies if not a.isPacman() and a.getPosition() is not None]
        walls = gameState.getWalls()
        x1, y1 = position1
        x2, y2 = position2

        if (walls[x1][y1] or enemyGhosts[x1][y1]):
            raise ValueError('Position1 is a wall: ' + str(position1))

        if (walls[x2][y2] or enemyGhosts[x2][y2]):
            raise ValueError('Position2 is a wall: ' + str(position2))

        prob = PositionSearchProblem(gameState, start = position1, goal = position2)

        return len(search.breadthFirstSearch(prob))
   
    def chooseAction(self, gameState):
        """
            problem = AnyFoodSearchProblem(gameState, gameState.getAgentPosition(self.index))
            heuristic = self.enemyDistanceHeuristic
            return self.aStarSearch(problem, gameState, heuristic)
        """
        """
        problem = AnyFoodSearchProblem(gameState, self.index)
        action = self.aStarSearch(problem, gameState, self.enemyDistanceHeuristic)[0]
        print("action: " , action)
        return action
        """
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
 
        return random.choice(bestActions)
        
    # Calculates the Euclidean distance from our agent's position to the nearest enemy agent's position.
    # This heuristic function is used for the h-value in A* Search Algorithm to avoid an enemy similar to wall collision.
    def enemyDistanceHeuristic(self, state, gameState):
    
        # Our agent's position.
        agentPosition = gameState.getAgentPosition(self.index)
        print("agent pos:  ", agentPosition)
        
        # The position of the enemy ghost with the lowest Euclidean distance from our agent.
        opponents = self.getOpponents(gameState)
        print("opp ", gameState.getAgentPosition(opponents[0]))
        enemyPosition =  gameState.getAgentPosition(opponents[0])# self.getNearestGhostPosition(agentPosition)
        
        # distance = euclidean(pos1, pos2)
        x1, y1 = agentPosition
        print("agent:  ", agentPosition)
        x2, y2 = enemyPosition
        print("enemy:  ", enemyPosition)
        heuristic = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 # euclidean(gameState, enemyPosition)
        
        return heuristic

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList]) # cheesyAStar was previously getMazeDistance
            features['distanceToFood'] = minDistance
            
        capsules = self.getCapsules(successor)
        if len(capsules) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minCapsuleDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
            if minCapsuleDistance == 0:
                features['distanceToCapsule'] = 0
            else:
                features['distanceToCapsule'] = 1 / (minCapsuleDistance + 1)
        else:
            features['distanceToCapsule'] = 100

        opponents = []
        opponents = self.getOpponents(gameState)
        minOpponentDist = float('inf')
        myPos = successor.getAgentState(self.index).getPosition()
        for opp in opponents:
            pos = successor.getAgentState(opp).getPosition()
            currentOpponentDist = self.getMazeDistance(myPos, pos)
            if (currentOpponentDist < minOpponentDist):
                minOpponentDist = currentOpponentDist
   
        
            if successor.getAgentState(opp).getScaredTimer() == 0:
                features['distanceToOpponent'] = 1 / minOpponentDist
            if successor.getAgentState(opp).getScaredTimer() > 0:
                 features['distanceToOpponent'] = -1000
        else:
            features['distanceToOpponent'] = 0
       
        if (action == Directions.STOP):
            features['stop'] = 1 

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            'distanceToCapsule': 100, # ADDED
            'distanceToOpponent': -3, # ADDED
            'stop': -100, # ADDED
            'reverse': -2, # ADDED
        }

    
class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
        
    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
 
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        #distance to middle boundary
        features['boundaryDistance'] = 0 # ADDED
        
        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            features['onDefense'] = 1 # ADDED
        else: # ADDED
            features['boundaryDistance'] = self.getBoundaryDistance(successor) # ADDED
            minDistanceToFood = min([self.getMazeDistance(myPos, food) for food in self.getFood(successor).asList()]) # ADDED
            features['distanceToFood'] = minDistanceToFood # ADDED

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getBoundaryDistance(self, gameState):
        x = int(gameState.getWalls().getWidth() / 2)
        y = int(gameState.getWalls().getHeight() / 2)
        values = []
        
        if self.red:
            x = x - 1

        for i in range(y):
            if not gameState.hasWall(x, y):
                values.append((x, y))

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        minDist = float('inf')

        for centervalue in values:
            distanceToCenter = self.getMazeDistance(myPos, centervalue)
            if distanceToCenter <= minDist:
                minDist = distanceToCenter
        
        return minDist
        
    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2,
            'distanceToFood': -1,
            'boundaryDistance': -10
        }