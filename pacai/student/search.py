from pacai.util.stack import Stack as stack
from pacai.util.priorityQueue import PriorityQueue as queue
from pacai.util.queue import Queue

"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

"""
problem.successorStates() --> list of verticies for next frontiers

"""


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """
    # declare Node
    head = [problem.startingState(), 0, None, None]
    node = head
    # reg stack decl.
    f = stack()
    f.push(head)
    visited_states = []
    while len(f) > 0:
        node = f.pop()
        if problem.isGoal(node[0]):
            # if goal found, trace path and make path list
            path = []
            while node[3] is not None:
                path = [node[2]] + path
                node = node[3]
            return path
        if node[0] not in visited_states:
            visited_states.append(node[0])
            for nextState, action, cost in problem.successorStates(node[0]):
                # Node Structure: [nextState, cost, action, parentNode]
                child = [nextState, cost, action, node]
                f.push(child)
    return False

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    """
    if problem's initial state is a goal then return empty path to initial state
    frontier ← a FIFO queue initially containing one path, for the problem's initial state
    reached ← a set of states; initially empty
    solution ← failure
    while frontier is not empty do
      parent ← the first node in frontier
      for child in successors(parent) do
        s ← child.state
        if s is a goal then
          return child
        if s is not in reached then
          add s to reached
          add child to the end of frontier
    return solution
    """

    if problem.isGoal(problem.startingState()):
        return []
    f = Queue()
    node = [problem.startingState(), 0, None, None]
    f.push(node)
    visited_states = []
    while len(f) > 0:
        node = f.pop()
        for nextState, action, cost in problem.successorStates(node[0]):
            child = [nextState, cost, action, node]
            s = nextState
            if problem.isGoal(nextState):
                path = [action]
                while node[3] is not None:
                    path = [node[2]] + path
                    node = node[3]
                return path
            elif s not in visited_states:
                visited_states.append(s)
                f.push(child)
    return False

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    """
    function UNIFORM-COST-SEARCH(problem) returns a solution, or failure
    if problem's initial state is a goal then return empty path to initial state
    frontier ← a priority queue ordered by pathCost, with a node for the initial state
    reached ← a table of {state: the best path that reached state}; initially empty
    solution ← failure
    while frontier is not empty and top(frontier) is cheaper than solution do
      parent ← pop(frontier)
      for child in successors(parent) do
        s ← child.state
        if s is not in reached or child is a cheaper path than reached[s] then
          reached[s] ← child
          add child to the frontier
          if child is a goal and is cheaper than solution then
            solution = child
    return solution
    """
    if problem.isGoal(problem.startingState()):
        return []
    f = queue()
    node = [problem.startingState(), 0, None, None]
    f.push(node, 0)
    visited_states = {problem.startingState(): 0}
    while not f.isEmpty():
        node = f.pop()
        for nextState, action, cost in problem.successorStates(node[0]):
            child = [nextState, cost + node[1], action, node]
            s = nextState
            if problem.isGoal(s):
                path = []
                node = child
                while node[3] is not None:
                    path = [node[2]] + path
                    node = node[3]
                return path
            if (s not in visited_states.keys()):
                visited_states[s] = child[1]
                f.push(child, child[1])

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    if problem.isGoal(problem.startingState()):
        return []
    f = queue()
    node = [problem.startingState(), 0, None, None]
    f.push(node, heuristic(problem.startingState(), problem))
    visited_states = {problem.startingState(): 999999}
    while not f.isEmpty():
        node = f.pop()
        if problem.isGoal(node[0]):
            path = []
            while node[3] is not None:
                path = [node[2]] + path
                node = node[3]
            return path
        if node[0] not in visited_states:
            visited_states[node[0]] = 999999
        for nextState, action, cost in problem.successorStates(node[0]):
            total_cost = cost + node[1]
            if (nextState not in visited_states) or (total_cost < visited_states[nextState]):
                visited_states[nextState] = total_cost
                child = [nextState, total_cost, action, node]
                f.push(child, heuristic(nextState, problem) + total_cost)
    return []
