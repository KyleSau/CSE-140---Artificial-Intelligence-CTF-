from pacai.agents.learning.value import ValueEstimationAgent
from decimal import Decimal as D

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        # A dictionary which holds the q-values for each state.
        self.values = {state: D(0) for state in mdp.getStates()}
        self.policy = {state: None for state in mdp.getStates()}
        self.qvalues = {}
        for i in range(iters):
            values_new = self.values.copy()
            for state in mdp.getStates():
                action_values = {}
                if mdp.isTerminal(state):
                    self.policy[state] = None
                    continue
                for action in mdp.getPossibleActions(state):
                    action_values[action] = D(0)
                    for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                        reward = mdp.getReward(state, action, nextState)
                        action_values[action] += D(prob) * (D(reward)
                            + D(discountRate) * self.values[nextState])
                values_new[state] = D(max(action_values.values()))
                for action, value in action_values.items():
                    self.qvalues[(state, action)] = D(value)
                self.policy[state] = max(action_values, key=action_values.get)
            self.values = values_new

    def getPolicy(self, state):
        return self.policy[state]

    def getQValue(self, state, action):
        return float(self.qvalues[(state, action)])

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return float(self.values[state])

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
