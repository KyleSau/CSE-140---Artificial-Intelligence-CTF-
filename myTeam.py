from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
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
            'distanceToFood': -1, # ADDED
            'boundaryDistance': -10 # ADDED
        }
