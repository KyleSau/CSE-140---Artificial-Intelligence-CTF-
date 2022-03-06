from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util.probability import flipCoin
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: Implementation of Q-Learning update of Q Values and retrieval
    of policy and Q Values.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.qvalues = {}
        # You can initialize Q-values here.

    def update(self, state, action, nextState, reward):
        """
        Updates Q Value table when rewarded on an action according to the following formula:
        Q(i + 1) (s, a) = (1 - a) * Q(i) (s, a) + a * (sample) where sample = Reward(s, a, s') +
        discount * max(Q(s', a'))
        """
        # (1 - a) * Q(i) (s, a)
        s_a = (state, action)
        Q_Val = (1 - self.getAlpha()) * self.getQValue(state, action)
        # a * (Reward(s, a, s') + discount * max(Q(s', a')))
        Q_Val += (self.getAlpha() * (reward
            + self.getDiscountRate() * self.getValue(nextState)))
        self.qvalues[s_a] = Q_Val

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.qvalues[(state, action)] if (state, action) in self.qvalues.keys()\
            else 0.0

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
        if len(self.getLegalActions(state)) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in self.getLegalActions(state)])

    def getAction(self, state):
        rand_ac = random.choice(self.getLegalActions(state))
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
        all_actions = list(self.getLegalActions(state))
        if len(all_actions) == 0:
            return None
        i = 0
        while i < len(all_actions):
            if self.getQValue(state, all_actions[i]) != self.getValue(state):
                del all_actions[i]
                continue
            i += 1
        return random.choice(all_actions)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)()
        self.weights = {}

    def update(self, state, action, nextState, reward):
        """
        Updates weights on transition according to the following formula:
        w(i + 1) = w(i) + alpha * correction * feature where
        correction = (reward * discount * V(s')) - Q(s, a)
        """
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
