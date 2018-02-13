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
import traceback

import numpy as np

from game import Agent


_DEBUG = False
ghost_weight = 1
capsule_weight = 1
score_weight = 0.1
scared_weight = 10

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
        # print('scores: {}'.format(scores))
        bestScore = max(scores)
        # print('best score: {}'.format(bestScore))
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        print(chosenIndex)

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        food_list = newFood.asList()
        ghost_list = [state.getPosition() for state in newGhostStates]

        if _DEBUG:
            print('\nTIMESTEP')
            print('--------\n')

            print("successor State:\n {}".format(successorGameState))
            print("new pos: {}".format(newPos))
            print("new Food: {}".format(newFood))
            print("Food: {}".format(food_list))
            print("new ghost states: {}".format(newGhostStates))
            print("ghost position: {}".format(ghost_list))
            print("new scared times: {}".format(newScaredTimes))


        # GET DISTANCES
        food_distances = []
        for food in food_list:
            dist = float(util.manhattanDistance(food, newPos))
            food_distances.append(dist)

        avg_food_distance = np.mean(food_distances)
        if len(food_distances) > 0:
            closest_food = min(food_distances)
        else:
            closest_food = 1

        food_sum = sum([1.0/float(dist) for dist in food_distances])


        ghost_distances = []
        for food in ghost_list:
            dist = util.manhattanDistance(food, newPos)
            ghost_distances.append(dist)

        avg_ghost_distance = np.mean(ghost_distances)
        closest_ghost = min(ghost_distances)
        if closest_ghost == 0:
             closest_ghost = 0.1


        # ghost_sum = sum([1.0/float(dist) for dist in ghost_distances])

        if _DEBUG:
            print("Food Distances: {}".format(food_distances))
            print("AVG Distances: {}".format(avg_food_distance))
            print("food sum: {}".format(food_sum))

            print("Ghost Distances: {}".format(ghost_distances))
            print("AVG Distances: {}".format(avg_ghost_distance))

            print("score: {}".format(successorGameState.getScore()))

        # new_score = int(food_sum*100)
        new_score = 1.0/closest_food - ghost_weight/closest_ghost + successorGameState.getScore()

        return new_score


        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

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
        "*** YOUR CODE HERE ***"

        legalActions = gameState.getLegalActions()
        bestAction = Directions.STOP
        bestScore = -(float("inf"))

        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            nextScore = self.getValue(nextState, 0, 1)
            if nextScore > bestScore:
                bestAction = action
                bestScore = nextScore
        return bestAction

    def maxValue(self, gameState, currentDepth):
        values = [float("-inf")]
        for action in gameState.getLegalActions(0):
            values.append(self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))

        return max(values)

    def minValue(self, gameState, currentDepth, agentIndex):
        values = [float("inf")]
        for action in gameState.getLegalActions(agentIndex):
            lastGhostIndex = gameState.getNumAgents() - 1
            if agentIndex == lastGhostIndex:
                values.append(self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0))
            else:
                values.append(self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1))

        return min(values)

    def getValue(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth)
        else:
            return self.minValue(gameState, currentDepth, agentIndex)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        legalActions = gameState.getLegalActions()
        bestAction = Directions.STOP
        bestScore = -(float("inf"))
        alpha = -(float("inf"))
        beta = float("inf")

        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            nextScore = self.getValue(nextState, 0, 1, alpha, beta)
            if nextScore > bestScore:
                bestAction = action
                bestScore = nextScore
            alpha = max(alpha,bestScore)
        return bestAction

    def maxValue(self, gameState, alpha, beta, currentDepth):
        #values = [float("-inf")]
        v = -(float("inf"))
        for action in gameState.getLegalActions(0):
            v = max(v,self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha,v)
        return v

        #return max(values)

    def minValue(self, gameState, alpha, beta, currentDepth, agentIndex):
        #values = [float("inf")]
        v = float("inf")
        for action in gameState.getLegalActions(agentIndex):
            lastGhostIndex = gameState.getNumAgents() - 1
            if agentIndex == lastGhostIndex:
                v = min(v,self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0, alpha, beta))
            else:
                v = min(v,self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta,v)
        return v
        #return min(values)

    def getValue(self, gameState, currentDepth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState, alpha, beta, currentDepth)
        else:
            return self.minValue(gameState, alpha, beta, currentDepth, agentIndex)
'''
        def maxValue(gameState, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = -(float("inf"))
            legalActions = gameState.getLegalActions(0) # maxValue will only be used for Pacman, always use index 0
            for action in legalActions:
                nextState = gameState.generateSuccessor(0, action)
                v = max(v, minValue(nextState, alpha, beta, 1, depth))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState, alpha, beta, agentIndex, depth):
            numberGhosts = gameState.getNumAgents()-1 # Gets the number of ghosts, assuming number of agents is pacman + ghosts
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = float("inf")
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                nextState = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == numberGhosts:
                    v = min(v, maxValue(nextState, alpha, beta, depth - 1)) # after all ghosts, reduce depth by 1
                    if v < alpha:
                        return v
                    beta = min(beta,v)
                else:
                    v = min(v, minValue(nextState, alpha, beta, agentIndex + 1, depth))
                    if v < alpha:
                        return v
                    beta = min(beta,v)
            return v

        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        bestScore = -(float("inf"))
        alpha = -(float("inf"))
        beta = float("inf")
        for action in legalActions:
            nextState = gameState.generateSuccessor(0,action)
            nextScore = max(bestScore,minValue(nextState,alpha,beta,1,self.depth))
            if nextScore > bestScore:
                bestAction = action
            if nextScore >= beta:
                return bestAction
            alpha = max(alpha,bestScore)
        return bestAction
'''
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
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
        "*** YOUR CODE HERE ***"

        legalActions = gameState.getLegalActions()
        bestAction = Directions.STOP
        bestScore = -(float("inf"))

        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            nextScore = self.getValue(nextState, 0, 1)
            if nextScore > bestScore:
                bestAction = action
                bestScore = nextScore
        return bestAction

    def maxValue(self, gameState, currentDepth):
        values = [float("-inf")]
        for action in gameState.getLegalActions(0):
            values.append(self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))

        return max(values)

    def minValue(self, gameState, currentDepth, agentIndex):
        values = []
        for action in gameState.getLegalActions(agentIndex):
            lastGhostIndex = gameState.getNumAgents() - 1
            if agentIndex == lastGhostIndex:
                values.append(self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth+1, 0))
            else:
                values.append(self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1))

        return float(sum(values))/float(len(values))

    def getValue(self, gameState, currentDepth, agentIndex):
        if gameState.isWin() or gameState.isLose() or currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.maxValue(gameState,currentDepth)
        else:
            return self.minValue(gameState, currentDepth, agentIndex)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newCapsules = currentGameState.getCapsules()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    food_list = newFood.asList()
    ghost_list = [state.getPosition() for state in newGhostStates]

    # GET DISTANCES
    food_distances = []
    for food in food_list:
        dist = float(util.manhattanDistance(food, newPos))
        food_distances.append(dist)

    avg_food_distance = np.mean(food_distances)
    if len(food_distances) > 0:
        closest_food = min(food_distances)
    else:
        closest_food = 1

    food_sum = sum([1.0/float(dist) for dist in food_distances])

    # Ghosts
    ghost_distances = []
    for food in ghost_list:
        dist = util.manhattanDistance(food, newPos)
        ghost_distances.append(dist)

    avg_ghost_distance = np.mean(ghost_distances)
    closest_ghost = min(ghost_distances)
    if closest_ghost == 0:
            closest_ghost = 0.1

    # Capsules
    capsule_distances = []
    for capsule in newCapsules:
        dist = util.manhattanDistance(capsule, newPos)
        capsule_distances.append(dist)

    avg_capsule_distance = np.mean(capsule_distances)
    if len(capsule_distances) == 0:
        closest_capsule = 1000
    else:
        closest_capsule = min(capsule_distances)

    scared_ghosts = max(newScaredTimes)
    if scared_ghosts > 0:
        scared_score = 1
    else:
        scared_score = 0

    new_score = 1.0/closest_food - ghost_weight/closest_ghost +  \
                capsule_weight/closest_capsule +  score_weight*currentGameState.getScore() + \
                scared_weight*scared_score

    return new_score


# Abbreviation
better = betterEvaluationFunction

