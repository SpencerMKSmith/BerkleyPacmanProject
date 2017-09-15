# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stackFringe = util.Stack()
    stackFringe.push((problem.getStartState(), [], 0, nullHeuristic, problem))
    return graphSearch(problem, stackFringe)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queueFringe = util.Queue()
    queueFringe.push((problem.getStartState(), [], 0, nullHeuristic, problem))
    return graphSearch(problem, queueFringe)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priorityQueueFringe = util.PriorityQueueWithFunction(ucsPriorityFunction)
    priorityQueueFringe.push((problem.getStartState(), [], 0, nullHeuristic, problem))
    return graphSearch(problem, priorityQueueFringe)

def ucsPriorityFunction(item):
    return item[2]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priorityQueueFringe = util.PriorityQueueWithFunction(aStarPriorityValueFunction)
    priorityQueueFringe.push((problem.getStartState(), [], 0, heuristic, problem))
    return graphSearch(problem, priorityQueueFringe)

def aStarPriorityValueFunction(item):
    currentState = item[0]
    costToHere = item[2]
    heuristicFunction = item[3]
    problem = item[4]
    hValue = heuristicFunction(currentState, problem)
    return costToHere + hValue

def graphSearch(problem, fringe):
    closedSet = []

    while(not fringe.isEmpty()):
        currentNode, actionsToHere, costToHere, hFunc, problem = fringe.pop()

        if problem.isGoalState(currentNode):
            return actionsToHere
        elif currentNode not in closedSet:
            closedSet += [currentNode]
            for successor, actionToNext, costToNext in problem.getSuccessors(currentNode):
                fringe.push((successor, actionsToHere + [actionToNext], costToHere + costToNext, hFunc, problem))




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
