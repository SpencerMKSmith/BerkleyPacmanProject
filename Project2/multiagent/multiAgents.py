# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Get the (x, y) list of where the ghosts will be
        newGhostLocations = [newGhostState.configuration.pos for newGhostState in newGhostStates]

        # Get the distances to the closest food and ghost
        distanceToClosestFood, closestFood = distanceToClosestNode(manhattanDistance, newPos, newFood)
        distanceToClosestGhost, closestGhost = distanceToClosestNode(manhattanDistance, newPos, newGhostLocations)
        
        # Set the initial value to the game state score
        evaluatedScore = successorGameState.getScore()
        
        # If we eat a pellet in this space, give 1000 points
        if newPos in oldFood: evaluatedScore += 1000
        
        # If the distance to the closest ghost is less than 2, give significant reduction, else only minor
        if(distanceToClosestGhost <= 2):
            evaluatedScore -= 2000 * ( 1 / (distanceToClosestGhost + 1) )
        else:
            evaluatedScore -= (1 / (100 * distanceToClosestGhost + 1))

        # Add points proportional to the closest food pellet
        evaluatedScore += 100 / (distanceToClosestFood + 1)
        
        return evaluatedScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

    
def euclideanDistance(xy1, xy2):
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    
# Creates a list of distances to nodes and return the shortest
def distanceToClosestNode( distanceFunction, currentNode, nodeList ):

    if len(nodeList) == 0:
        return (1, currentNode)
        
    distanceToNodes = [(distanceFunction(currentNode, node), node) for node in nodeList]
    return min(distanceToNodes, key = lambda t: t[0])
    
class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        bestAction, bestScore = self.MaxValue(gameState, -1)
        return bestAction

    def MaxValue(self, gameState, thisDepth):
        thisDepth += 1

        # If we are at the lowest depth then return some arbitrary action with the value of the state
        atLowestDepth = (thisDepth >= self.depth)
        if atLowestDepth or gameState.isWin() or gameState.isLose(): return ("Stop", self.evaluationFunction(gameState))

        maxAction = ""
        maxValue = float("-inf")

        for legalAction in gameState.getLegalActions(0):
            nextActionState = gameState.generateSuccessor(0, legalAction)

            # Get the value of the minimizers after taking the action
            stateValue = self.MinValue(nextActionState, thisDepth, 1)
            if stateValue > maxValue:
                maxAction = legalAction
                maxValue = stateValue

        # Returns a tuple of the best action to take and the value of the path
        return (maxAction, maxValue)
        
    def MinValue(self, gameState, thisDepth, ghostIndex):
        
        isLastGhost = ghostIndex == (gameState.getNumAgents() -1)
        atLowestDepth = (thisDepth == self.depth)

        # Check if is maximum depth of terminal state
        if atLowestDepth or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)
        
        minValue = float("inf")
        
        for legalAction in gameState.getLegalActions(ghostIndex):
            nextActionState = gameState.generateSuccessor(ghostIndex, legalAction)

            # If it is the last ghost, call the Max function
            if isLastGhost:
                minValue = min(minValue, self.MaxValue(nextActionState, thisDepth)[1])
            else: # Else call the min function for the next ghost
                minValue = min(minValue, self.MinValue(nextActionState, thisDepth, ghostIndex + 1))
        
        return minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float("-inf")
        beta = float("inf")
        bestAction, bestScore = self.MaxValuePrune(gameState, -1, alpha, beta)
        return bestAction

    def MaxValuePrune(self, gameState, thisDepth, alpha, beta):
        thisDepth += 1

        # If we are at the lowest depth then return some arbitrary action with the value of the state
        atLowestDepth = (thisDepth >= self.depth)
        if atLowestDepth or gameState.isWin() or gameState.isLose(): return ("Stop", self.evaluationFunction(gameState))

        maxAction = ""
        maxValue = float("-inf")

        for legalAction in gameState.getLegalActions(0):
            nextActionState = gameState.generateSuccessor(0, legalAction)

            # Get the value of the minimizers after taking the action
            stateValue = self.MinValuePrune(nextActionState, thisDepth, 1, alpha, beta)
            if stateValue > maxValue:
                maxAction = legalAction
                maxValue = stateValue

            # Set the new alpha value if this is higher than the current
            # Prune if this state value is greater than the beta, meaning that the minimizer would never choose
            #   the producer of this branch.  NOTE: Doesn't matter what action we return, it won't be chosen.
            alpha = max(alpha, stateValue)
            if stateValue > beta: return ("Stop", maxValue)

        # Returns a tuple of the best action to take and the value of the path
        return (maxAction, maxValue)

    def MinValuePrune(self, gameState, thisDepth, ghostIndex, alpha, beta):

        isLastGhost = ghostIndex == (gameState.getNumAgents() - 1)
        atLowestDepth = (thisDepth == self.depth)

        # Check if is maximum depth of terminal state
        if atLowestDepth or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState)

        minValue = float("inf")

        for legalAction in gameState.getLegalActions(ghostIndex):
            nextActionState = gameState.generateSuccessor(ghostIndex, legalAction)

            # If it is the last ghost, call the Max function
            if isLastGhost:
                minValue = min(minValue, self.MaxValuePrune(nextActionState, thisDepth, alpha, beta)[1])
            else:  # Else call the min function for the next ghost
                minValue = min(minValue, self.MinValuePrune(nextActionState, thisDepth, ghostIndex + 1, alpha, beta))

            # Set the new beta value if it's less than
            # If the value is below the current alpha then return because the maximizer will never
            #   choose this branch
            beta = min(beta, minValue)
            if minValue < alpha: return minValue

        return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    return currentGameState.getScore();

# Abbreviation
better = betterEvaluationFunction

