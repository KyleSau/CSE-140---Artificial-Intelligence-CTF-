from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util.priorityQueue import PriorityQueue
from pacai.agents.search.base import SearchAgent
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

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
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
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }
        
    def getEnemyDistances(self, gameState):
        # Puts all the distances of the enemies in a list.
        # ...
        return 0
        
    def getNearestEnemyDistance(self, gameState):
        return min(getEnemyDistances(gameState))
        
    def enemyDistanceHeuristic(self, state, gameState):
        # Calculates the Euclidean distance from our agent's position to the nearest enemy agent's position.
        # This heuristic function is used for the h-value in A* Search Algorithm to avoid an enemy identical to collision.
        
        #agentPosition
        #enemyPosition
        
        #distance = euclidean(agentPosition, enemyPosition)
        heuristic = euclidean(agentPosition, enemyPosition)
        return heuristic
    
    def aStarSearch(self, problem, gameState, heuristic):
        start_state = problem.getStartState()
        fringe = PriorityQueue()
        h = heuristic(start_state, gameState)
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
            successors = problem.getSuccessors(state)
            for successor in successors:
              current_path = list(path)
              successor_state = successor[0]
              move = successor[1]
              g = successor[2] + current_cost
              h = heuristic(successor_state, gameState)
              if successor_state not in explored:
                current_path.append(move)
                f = g + h
                successor_node = (successor_state, current_path, g)
                fringe.push(successor_node, f)
        return []
    
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
