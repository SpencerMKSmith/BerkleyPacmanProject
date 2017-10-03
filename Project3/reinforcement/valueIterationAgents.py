# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util, sys

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # #We want to code the value iteration algorithm here
        for value in range(self.iterations): #Loop through the number of required iterations
            valueCopy = dict(self.values) 
            for state in self.mdp.getStates(): 

                #Try and find the best action based on the q-value of the current action in current state
                bestStateValue = -1 * (sys.maxint)
                for action in self.mdp.getPossibleActions(state):
                    bestStateValue = max(bestStateValue, self.computeQValueFromValues(state, action))

                #Account for states that have no legal actions
                if (bestStateValue == (-1 * (sys.maxint))):
                    bestStateValue = 0
                valueCopy[state] = bestStateValue
            self.values = valueCopy

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """

        bestQValue = 0
        for nextState, transitionProbability in self.mdp.getTransitionStatesAndProbs(state, action):
            bestQValue += transitionProbability * (self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState])
        return bestQValue
          
    def computeActionFromValues(self, state):
      """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
      """

      #Store the best STATE & ACTION
      bestActionPair = [-1 * sys.maxint, None]
      for action in self.mdp.getPossibleActions(state):
          currentValue = self.computeQValueFromValues(state, action)
          if (currentValue > bestActionPair[0]):
              bestActionPair[0] = currentValue
              bestActionPair[1] = action
      return bestActionPair[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)