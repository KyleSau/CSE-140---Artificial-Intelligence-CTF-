from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import util
from pacai.core.directions import Directions
import random
from pacai.util.probability import flipCoin
from pacai.core.actions import Actions
from pacai.core.search import search
from pacai.core.search.position import PositionSearchProblem
from pacai.core.distance import maze, euclidean
from pacai.student.search import breadthFirstSearch
import pickle
from os.path import exists


def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        TeamAgent(firstIndex, 0),
        secondAgent(secondIndex),
    ]

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.weights = {}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        return self.getAction(gameState)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        successor = self.getSuccessor(gameState, action)

        return {
            'successorScore': self.getScore(successor)
        }

    def update(self, state, action, reward):
        """
        Updates weights on transition according to the following formula:
        w(i + 1) = w(i) + alpha * correction * feature where
        correction = (reward * discount * V(s')) - Q(s, a)
        """
        nextState = self.getSuccessor(state, action)
        for feat, f in self.featExtractor.getFeatures(state, action).items():
            if feat not in self.weights:
                self.weights[feat] = 0.0
            correction = reward + self.getDiscountRate() * self.getValue(nextState)
            correction -= self.getQValue(state, action)
            self.weights[feat] += self.getAlpha() * correction * f

    def getQValue(self, state, action):
        """
        Get the Q-Value for a state, action pair.
        """
        return sum([self.weights[feature] * f if feature in self.weights else
                    0.0 for feature, f in
                    self.featExtractor.getFeatures(state, action).items()])

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            print("Final Weights: ")
            for state_action, weight in self.weights.items():
                print("{}    {}".format(state_action, weight))

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        if len(state.getLegalActions(self.index)) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in state.getLegalActions(self.index)])

    def getAction(self, state):
        rand_ac = random.choice(state.getLegalActions(self.index))
        return rand_ac if flipCoin(self.getEpsilon()) else self.getPolicy(state)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        all_actions = list(state.getLegalActions(self.index))
        if len(all_actions) == 0:
            return None
        i = 0
        while i < len(all_actions):
            if self.getQValue(state, all_actions[i]) != self.getValue(state):
                del all_actions[i]
                continue
            i += 1
        return random.choice(all_actions)

class TeamAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, whichAgent, **kwargs):
        super().__init__(index, **kwargs)
        self.whichAgent = whichAgent
        self.epsilon = 0.2
        self.gamma = 0.95
        self.alpha = 0.05
        self.weights  ={}
        if exists('weights.pickle'):
            with open('weights.pickle', 'rb') as f:
                self.weights = pickle.load(f)
        # self.weights = {'bias'  :  7.24187586664831,
        # '#-of-ghosts-1-step-away'  :  -8.081741961287042,
        # 'closest-food' :   -0.32191107412668146,
        # 'eats-food'  :  6.934986318078202,
        # 'scared-ghost-timer'  :  3.61021620382729,
        # 'closest-capsule'  :  0.0048304370972912965,
        # 'eats-capsule' :   0.13602267844540306,
        # 'closest-scared-ghost'  :  13.946460551342293}
        self.qvalues = {}
        self.statehist = []

    def getEpsilon(self):
        return self.epsilon

    def getDiscountRate(self):
        return self.gamma

    def getAlpha(self):
        return self.alpha

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """
        return self.getAction(gameState)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        successor = self.getSuccessor(gameState, action)

        return {
            'successorScore': self.getScore(successor)
        }

    def update(self, state, action, reward):
        """
        Updates weights on transition according to the following formula:
        w(i + 1) = w(i) + alpha * correction * feature where
        correction = (reward * discount * V(s')) - Q(s, a)
        """
        nextState = self.getSuccessor(state, action)
        for feat, f in self.getFeatures(state, action).items():
            if feat not in self.weights:
                self.weights[feat] = 0.0
            correction = reward + self.getDiscountRate() * self.getValue(nextState)
            correction -= self.getQValue(state, action)
            self.weights[feat] += self.getAlpha() * correction * f

    def getQValue(self, state, action):
        """
        Get the Q-Value for a state, action pair.
        """
        return sum([self.weights[feature] * f if feature in self.weights else
                    0.0 for feature, f in
                    self.getFeatures(state, action).items()])

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        print("Final Weights: ")
        for state_action, weight in self.weights.items():
            print("{}    {}".format(state_action, weight))

        with open('weights.pickle', 'wb') as f:
            pickle.dump(self.weights, f, protocol=pickle.HIGHEST_PROTOCOL)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        if len(state.getLegalActions(self.index)) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in state.getLegalActions(self.index)])

    def getAction(self, state):
        rand_ac = random.choice(state.getLegalActions(self.index))
        ret_action = rand_ac
        #if scared ghost bla bla
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        scaredGhosts = {a.getPosition(): a.getScaredTimer() for a in enemies if a.getScaredTimer() > 0}
        if len(self.statehist) > 14:
            del self.statehist[0]
        if len(scaredGhosts.keys()) > 0:
            ret_action = breadthFirstSearch(PositionSearchProblem(state, start=state.getAgentState(self.index).getPosition(), goal=min(scaredGhosts, key=scaredGhosts.get)))[0]
            self.statehist.append(self.getSuccessor(state, ret_action).getAgentPosition(self.index))
            return ret_action
        if flipCoin(self.getEpsilon()):
            self.statehist.append(self.getSuccessor(state, rand_ac).getAgentPosition(self.index))
            return rand_ac
        else:
            ret_action = self.getPolicy(state)
            self.statehist.append(self.getSuccessor(state, ret_action).getAgentPosition(self.index))
            return ret_action

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        all_actions = list(state.getLegalActions(self.index))
        if len(all_actions) == 0:
            return None
        i = 0
        while i < len(all_actions):
            self.update(state, all_actions[i], self.getReward(state, all_actions[i]))
            if self.getQValue(state, all_actions[i]) != self.getValue(state):
                del all_actions[i]
                continue
            i += 1
        if state.getAgentPosition(self.index)[0] < ((state.getWalls().getWidth() / 2) - 1):# add check for red or blue team
            foodpos = {food:self.getMazeDistance(state.getAgentState(self.index).getPosition(), food) for food in self.getFood(state).asList()}
            ret_action = breadthFirstSearch(PositionSearchProblem(state, start=state.getAgentState(self.index).getPosition(), goal=min(foodpos, key=foodpos.get)))[0]
            return ret_action
        if self.isStuck():
            foodpos = {food:self.getMazeDistance(state.getAgentState(self.index).getPosition(), food) for food in self.getFood(state).asList()}
            ret_action = breadthFirstSearch(PositionSearchProblem(state, start=state.getAgentState(self.index).getPosition(), goal=min(foodpos, key=foodpos.get)))[0]
            return ret_action
        if len(all_actions) == 0:
            return random.choice(state.getLegalActions(self.index))
        return random.choice(all_actions)
    def getReward(self, state, action):
        nextState = self.getSuccessor(state, action)
        #if pacman in the nextstate will be eaten, we want a negative reward
        enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
        reward = (nextState.getScore() - state.getScore())
        if self.getSuccessor(state, action).getAgentPosition(self.index) in state.getCapsules():
            reward = 0.8
        death_wish_if_proceed = [a for a in enemies if not a.isPacman() and (not a.isScared()) and self.getMazeDistance(a.getPosition(), nextState.getAgentPosition(self.index)) <= 1]
        if len(death_wish_if_proceed) > 0:
            reward = -1
        # scared_ghosts = [a for a in enemies if not a.isPacman() and  a.isScared() and self.getMazeDistance(a.getPosition(), nextState.getAgentPosition(self.index)) <= 1]
        # if len(scared_ghosts) > 0:
        #     print('scared ghost')
        #     reward = 1.5
        return reward

    def getGhostPositions(self, state, action):
        #only caviet is its only visible ghosts we can get locations of
        successor = self.getSuccessor(state, action)
        opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        return [a.getPosition() for a in opponents if not a.isPacman() and a.getPosition() is not None]

    def getFeatures(self, state, action):
        # Extract the grid of food and wall locations and get the ghost locations.
        food = state.getFood()
        capsule = state.getCapsules()
        walls = state.getWalls()
        ghosts = self.getGhostPositions(state, action)

        features = {}
        features["bias"] = 1.0
        #features["successorScore"] = self.getScore(self.getSuccessor(state, action))
        # Compute the location of pacman after he takes the action.
        next_x, next_y = self.getSuccessor(state, action).getAgentPosition(self.index)

        # Count the number of ghosts 1-step away.
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
                Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # If there is no danger of ghosts then add the food feature.
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        # If there is no danger of ghosts then add the capsule feature.
        if not features["#-of-ghosts-1-step-away"] and (next_x,next_y) in capsule:
            features["eats-capsule"] = 1.0 #maybe we want to remove bc we wanna eat capsule so we can eat ghost? idk how that works

        # enemies = [self.getSuccessor(state, action).getAgentState(i) for i in self.getOpponents(self.getSuccessor(state, action))]
        # scaredGhosts = {a.getPosition(): a.getScaredTimer() for a in enemies}
        # features['scared-ghost-timer'] = min(scaredGhosts.values()) / 40#divide by 40 should normalize I think

        # if features['scared-ghost-timer'] > 0:
        #     features['closest-scared-ghost'] = min([self.getMazeDistance((next_x, next_y), scared) for scared in list(scaredGhosts.keys())])

        foods = [self.getMazeDistance((next_x, next_y), foodz) for foodz in food.asList()]
        dist = min(foods)
        if dist is not None:
            # Make the distance a number less than one otherwise the update will diverge wildly.
            features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

        capsules = [self.getMazeDistance((next_x, next_y), capsulez) for capsulez in capsule]

        capsuledist = min(capsules) if len(capsules) > 0 else None
        if capsuledist is not None:
            # Make the distance a number less than one otherwise the update will diverge wildly.
            features["closest-capsule"] = float(capsuledist) / (walls.getWidth() * walls.getHeight())

        for key in features:
            features[key] /= 10.0

        return features
    def isStuck(self):
        actionlist = {state:self.statehist.count(state) for state in self.statehist}
        countgreaterthan2 = 0
        for state, count in actionlist.items():
            if count > 3:
                countgreaterthan2 += 1
        return True if countgreaterthan2 > 1 else False