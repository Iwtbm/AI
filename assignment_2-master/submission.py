# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math
# import time


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.index = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        pop_node = heapq.heappop(self.queue)

        return (pop_node[0], pop_node[2])

        # raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        # raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        new_node = [node[0], self.index, node[1]]
        heapq.heappush(self.queue, new_node)
        self.index += 1

        
        # raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]



def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if goal == start:
        return []

    tree = PriorityQueue()
    discover_dict={}
    index = 0
    tree.append(((index, start),0))
    discover_dict[start] = start
    path = []

    while tree.size()>0:
        index += 1
        checking = tree.pop()

        for k in sorted(graph[checking[0][1]]):
            if k not in discover_dict:
                if k == goal:
                    trace_back = checking[0][1]
                    path.append(goal)
                    path.append(trace_back)

                    while trace_back != start:
                        path.append(discover_dict[trace_back])
                        trace_back = discover_dict[trace_back]

                    path.reverse()
                    return path

                tree.append(((index, k),0))
                discover_dict[k] = checking[0][1]

    raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if goal == start:
        return []
    
    tree = PriorityQueue()
    discover_dict={}
    tree.append((0, start))
    discover_dict[start] = [start, 0]
    path = []

    while tree.size()>0:
        checking = tree.pop()
        if checking[0] != discover_dict[checking[1]][1]:
            continue

        if checking[1] == goal:
            trace_back = checking[1]
            path.append(trace_back)

            while trace_back != start:
                path.append(discover_dict[trace_back][0])
                trace_back = discover_dict[trace_back][0]

            path.reverse()
            return path

        for k in graph[checking[1]]:
            child_distance = graph.get_edge_weight(checking[1], k) + checking[0]

            if k not in discover_dict:
                discover_dict[k] = [checking[1], child_distance]
                tree.append((child_distance, k))

            elif child_distance < discover_dict[k][1]:
                tree.append((child_distance, k))
                discover_dict[k] = [checking[1], child_distance]

    raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    p1 = graph.nodes[v]['pos']
    p2 = graph.nodes[goal]['pos']
    d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    return d
    # raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if goal == start:
        return []
    
    tree = PriorityQueue()
    discover_dict={}
    d_start = heuristic(graph, start, goal)
    current_d = 0
    tree.append((d_start, start))
    discover_dict[start] = [start, d_start, current_d]
    path = []

    while tree.size()>0:
        checking = tree.pop()
        if checking[0] != discover_dict[checking[1]][1]:
            continue

        if checking[1] == goal:
            trace_back = checking[1]
            path.append(trace_back)
            while trace_back != start:
                path.append(discover_dict[trace_back][0])
                trace_back = discover_dict[trace_back][0]

            path.reverse()
            return path

        for k in graph[checking[1]]:
            current_d = graph.get_edge_weight(checking[1], k) + discover_dict[checking[1]][2]
            child_distance = current_d + heuristic(graph, k, goal)

            if k not in discover_dict:
                discover_dict[k] = [checking[1], child_distance, current_d]
                tree.append((child_distance, k))

            elif child_distance < discover_dict[k][1]:
                tree.append((child_distance, k))
                discover_dict[k] = [checking[1], child_distance, current_d]

    raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if goal == start:
        return []
    
    tree_s = PriorityQueue()
    discover_dict_s = {}
    tree_s.append((0, start))
    discover_dict_s[start] = [start, 0]

    tree_g = PriorityQueue()
    discover_dict_g = {}
    tree_g.append((0, goal))
    discover_dict_g[goal] = [goal, 0]
    scaned_dict_g = {}
    scaned_dict_s = {}
    mu = float('inf')
    path_s = []
    path_g = []
    path = []

    while tree_s.size() > 0 and tree_g.size() > 0:
        checking_s = tree_s.pop()
        checking_g = tree_g.pop()

        while checking_s[0] != discover_dict_s[checking_s[1]][1]:
            checking_s = tree_s.pop()

        while checking_g[0] != discover_dict_g[checking_g[1]][1]:
            checking_g = tree_g.pop()

        scaned_dict_g[checking_g[1]] = 0
        scaned_dict_s[checking_s[1]] = 0

        if checking_g[0] + checking_s[0] >= mu:
            trace_back = intersect_point
            path_s.append(trace_back)

            while trace_back != start:
                path_s.append(discover_dict_s[trace_back][0])
                trace_back = discover_dict_s[trace_back][0]
            path_s.reverse()

            trace_back = intersect_point
            while trace_back != goal:
                path_g.append(discover_dict_g[trace_back][0])
                trace_back = discover_dict_g[trace_back][0]

            path = path_s + path_g
            return path

        for k in graph[checking_s[1]]:
            child_distance = graph.get_edge_weight(checking_s[1], k) + checking_s[0]

            if k not in discover_dict_s:
                discover_dict_s[k] = [checking_s[1], child_distance]
                tree_s.append((child_distance, k))

            elif child_distance < discover_dict_s[k][1]:
                tree_s.append((child_distance, k))
                discover_dict_s[k] = [checking_s[1], child_distance]

            if k in scaned_dict_g:
                current_path = discover_dict_s[k][1] + discover_dict_g[k][1]
                if mu > current_path: 
                    mu = current_path
                    intersect_point = k

        for k in graph[checking_g[1]]:
            child_distance = graph.get_edge_weight(checking_g[1], k) + checking_g[0]

            if k not in discover_dict_g:
                discover_dict_g[k] = [checking_g[1], child_distance]
                tree_g.append((child_distance, k))

            elif child_distance < discover_dict_g[k][1]:
                tree_g.append((child_distance, k))
                discover_dict_g[k] = [checking_g[1], child_distance]

            if k in scaned_dict_s:
                current_path = discover_dict_s[k][1] + discover_dict_g[k][1]
                if mu > current_path: 
                    mu = current_path
                    intersect_point = k

    raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if goal == start:
        return []

    pr_g = 0.5*(heuristic(graph, goal, start) - heuristic(graph, goal, goal)) + 0.5*heuristic(graph, start, goal)
    pf_s = 0.5*(heuristic(graph, start, goal) - heuristic(graph, start, start)) + 0.5*heuristic(graph, goal, start)
    p_f = 0.5*(heuristic(graph, start, goal) - heuristic(graph, start, start)) + 0.5*heuristic(graph, goal, start)
    p_r = 0.5*(heuristic(graph, goal, start) - heuristic(graph, goal, goal)) + 0.5*heuristic(graph, start, goal)
    current_d_s = 0
    current_d_g = 0

    tree_s = PriorityQueue()
    discover_dict_s = {}
    tree_s.append((p_f, start))
    discover_dict_s[start] = [start, current_d_s + p_f, current_d_s]

    tree_g = PriorityQueue()
    discover_dict_g = {}
    tree_g.append((p_r, goal))
    discover_dict_g[goal] = [goal, current_d_g + p_r, current_d_g]

    scaned_dict_g = {}
    scaned_dict_s = {}
    mu = float('inf')
    path_s = []
    path_g = []
    path = []

    while tree_s.size() > 0 and tree_g.size() > 0:
        checking_s = tree_s.pop()
        checking_g = tree_g.pop()

        while (checking_s[0] != discover_dict_s[checking_s[1]][1]) or (checking_s[0] in scaned_dict_s):
            checking_s = tree_s.pop()

        while (checking_g[0] != discover_dict_g[checking_g[1]][1]) or (checking_g[0] in scaned_dict_g):
            checking_g = tree_g.pop()

        scaned_dict_g[checking_g[1]] = 0
        scaned_dict_s[checking_s[1]] = 0

        if checking_g[0] + checking_s[0] >= mu + pr_g:
            trace_back = intersect_point
            path_s.append(trace_back)
            
            while trace_back != start:
                path_s.append(discover_dict_s[trace_back][0])
                trace_back = discover_dict_s[trace_back][0]
            path_s.reverse()

            trace_back = intersect_point
            while trace_back != goal:
                path_g.append(discover_dict_g[trace_back][0])
                trace_back = discover_dict_g[trace_back][0]

            path = path_s + path_g
            return path

        for k in graph[checking_s[1]]:
            if k in scaned_dict_s:
                continue

            p_f = 0.5*(heuristic(graph, k, goal) - heuristic(graph, k, start)) + 0.5*heuristic(graph, goal, start)
            current_d_s = graph.get_edge_weight(checking_s[1], k) + discover_dict_s[checking_s[1]][2]
            child_distance = current_d_s + p_f

            if k not in discover_dict_s:
                discover_dict_s[k] = [checking_s[1], child_distance, current_d_s]
                tree_s.append((child_distance, k))

            elif child_distance < discover_dict_s[k][1]:
                tree_s.append((child_distance, k))
                discover_dict_s[k] = [checking_s[1], child_distance, current_d_s]
            # if k in discover_dict_g:
            if k in scaned_dict_g:
                current_path = discover_dict_s[k][2] + discover_dict_g[k][2]
                if mu > current_path: 
                    mu = current_path
                    intersect_point = k

        for k in graph[checking_g[1]]:
            if k in scaned_dict_g:
                continue

            p_r = 0.5*(heuristic(graph, k, start) - heuristic(graph, k, goal)) + 0.5*heuristic(graph, start, goal)
            current_d_g = graph.get_edge_weight(checking_g[1], k) + discover_dict_g[checking_g[1]][2]
            child_distance = current_d_g + p_r

            if k not in discover_dict_g:
                discover_dict_g[k] = [checking_g[1], child_distance, current_d_g]
                tree_g.append((child_distance, k))

            elif child_distance < discover_dict_g[k][1]:
                tree_g.append((child_distance, k))
                discover_dict_g[k] = [checking_g[1], child_distance, current_d_g]
            # if k in discover_dict_s:
            if k in scaned_dict_s:
                current_path = discover_dict_s[k][2] + discover_dict_g[k][2]
                if mu > current_path: 
                    mu = current_path
                    intersect_point = k
    
    # raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    targets = set(goals)
    targets = list(targets)

    if len(targets) == 1:
        return []

    elif len(targets) == 2:
        start = targets[0]
        goal = targets[1]

        tree_s = PriorityQueue()
        discover_dict_s = {}
        tree_s.append((0, start))
        discover_dict_s[start] = [start, 0]

        tree_g = PriorityQueue()
        discover_dict_g = {}
        tree_g.append((0, goal))
        discover_dict_g[goal] = [goal, 0]
        scaned_dict_g = {}
        scaned_dict_s = {}
        mu = float('inf')
        path_s = []
        path_g = []
        path = []

        while tree_s.size() > 0 and tree_g.size() > 0:
            checking_s = tree_s.pop()
            checking_g = tree_g.pop()

            while checking_s[0] != discover_dict_s[checking_s[1]][1]:
                checking_s = tree_s.pop()

            while checking_g[0] != discover_dict_g[checking_g[1]][1]:
                checking_g = tree_g.pop()

            scaned_dict_g[checking_g[1]] = 0
            scaned_dict_s[checking_s[1]] = 0

            if checking_g[0] + checking_s[0] >= mu:
                trace_back = intersect_point
                path_s.append(trace_back)

                while trace_back != start:
                    path_s.append(discover_dict_s[trace_back][0])
                    trace_back = discover_dict_s[trace_back][0]
                path_s.reverse()

                trace_back = intersect_point
                while trace_back != goal:
                    path_g.append(discover_dict_g[trace_back][0])
                    trace_back = discover_dict_g[trace_back][0]

                path = path_s + path_g
                return path

            for k in graph[checking_s[1]]:
                child_distance = graph.get_edge_weight(checking_s[1], k) + checking_s[0]

                if k not in discover_dict_s:
                    discover_dict_s[k] = [checking_s[1], child_distance]
                    tree_s.append((child_distance, k))

                elif child_distance < discover_dict_s[k][1]:
                    tree_s.append((child_distance, k))
                    discover_dict_s[k] = [checking_s[1], child_distance]

                if k in scaned_dict_g:
                    current_path = discover_dict_s[k][1] + discover_dict_g[k][1]
                    if mu > current_path: 
                        mu = current_path
                        intersect_point = k

            for k in graph[checking_g[1]]:
                child_distance = graph.get_edge_weight(checking_g[1], k) + checking_g[0]

                if k not in discover_dict_g:
                    discover_dict_g[k] = [checking_g[1], child_distance]
                    tree_g.append((child_distance, k))

                elif child_distance < discover_dict_g[k][1]:
                    tree_g.append((child_distance, k))
                    discover_dict_g[k] = [checking_g[1], child_distance]

                if k in scaned_dict_s:
                    current_path = discover_dict_s[k][1] + discover_dict_g[k][1]
                    if mu > current_path: 
                        mu = current_path
                        intersect_point = k
    else:
        tree_s1 = PriorityQueue()
        discover_dict_s1 = {}
        tree_s1.append((0, targets[0]))
        discover_dict_s1[targets[0]] = [targets[0], 0]

        tree_s2 = PriorityQueue()
        discover_dict_s2 = {}
        tree_s2.append((0, targets[1]))
        discover_dict_s2[targets[1]] = [targets[1], 0]

        tree_s3 = PriorityQueue()
        discover_dict_s3 = {}
        tree_s3.append((0, targets[2]))
        discover_dict_s3[targets[2]] = [targets[2], 0]

        scaned_dict_s1 = {}
        scaned_dict_s2 = {}
        scaned_dict_s3 = {}

        mu1 = float('inf')
        mu2 = float('inf')
        mu3 = float('inf')

        continue1 = True
        continue2 = True
        continue3 = True

        path_s = []
        path_g = []
        path_list = []

        while tree_s1.size() > 0 and tree_s2.size() > 0 and tree_s3.size() > 0:
            if len(path_list) == 2:
                if continue1:
                    if mu1 > mu2 and mu1 > mu3:
                        break
                if continue2:
                    if mu2 > mu1 and mu2 > mu3:
                        break
                if continue3:
                    if mu3 > mu1 and mu3 > mu2:
                        break

            if len(path_list) == 3:
                break

            checking_s1 = tree_s1.pop()
            checking_s2 = tree_s2.pop()
            checking_s3 = tree_s3.pop()

            while checking_s1[0] != discover_dict_s1[checking_s1[1]][1] or (checking_s1[1] in scaned_dict_s1):
                checking_s1 = tree_s1.pop()

            while checking_s2[0] != discover_dict_s2[checking_s2[1]][1] or (checking_s2[1] in scaned_dict_s2):
                checking_s2 = tree_s2.pop()

            while checking_s3[0] != discover_dict_s3[checking_s3[1]][1] or (checking_s3[1] in scaned_dict_s3):
                checking_s3 = tree_s3.pop()

            scaned_dict_s1[checking_s1[1]] = 0
            scaned_dict_s2[checking_s2[1]] = 0
            scaned_dict_s3[checking_s3[1]] = 0

            if checking_s1[0] + checking_s2[0] >= mu1 and continue1:
                trace_back = intersect_point_1
                path_s.append(trace_back)

                while trace_back != targets[0]:
                    path_s.append(discover_dict_s1[trace_back][0])
                    trace_back = discover_dict_s1[trace_back][0]
                path_s.reverse()

                trace_back = intersect_point_1
                while trace_back != targets[1]:
                    path_g.append(discover_dict_s2[trace_back][0])
                    trace_back = discover_dict_s2[trace_back][0]

                path = path_s + path_g
                distance = discover_dict_s1[intersect_point_1][1] + discover_dict_s2[intersect_point_1][1]

                if targets[2] in path:
                    return path

                path_list.append((distance, path))
                continue1 = False
                path_s = []
                path_g = []

            if checking_s2[0] + checking_s3[0] >= mu2 and continue2:
                trace_back = intersect_point_2
                path_s.append(trace_back)

                while trace_back != targets[1]:
                    path_s.append(discover_dict_s2[trace_back][0])
                    trace_back = discover_dict_s2[trace_back][0]
                path_s.reverse()

                trace_back = intersect_point_2
                while trace_back != targets[2]:
                    path_g.append(discover_dict_s3[trace_back][0])
                    trace_back = discover_dict_s3[trace_back][0]

                path = path_s + path_g
                distance = discover_dict_s2[intersect_point_2][1] + discover_dict_s3[intersect_point_2][1]
                if targets[0] in path:
                    return path

                path_list.append((distance, path))
                continue2 = False
                path_s = []
                path_g = []

            if checking_s1[0] + checking_s3[0] >= mu3 and continue3:
                trace_back = intersect_point_3
                path_s.append(trace_back)

                while trace_back != targets[0]:
                    path_s.append(discover_dict_s1[trace_back][0])
                    trace_back = discover_dict_s1[trace_back][0]
                path_s.reverse()

                trace_back = intersect_point_3
                while trace_back != targets[2]:
                    path_g.append(discover_dict_s3[trace_back][0])
                    trace_back = discover_dict_s3[trace_back][0]

                path = path_s + path_g
                distance = discover_dict_s1[intersect_point_3][1] + discover_dict_s3[intersect_point_3][1]

                if targets[1] in path:
                    return path

                path_list.append((distance, path))
                continue3 = False
                path_s = []
                path_g = []

            for k in graph[checking_s1[1]]:
                if k in scaned_dict_s1:
                    continue

                child_distance = graph.get_edge_weight(checking_s1[1], k) + checking_s1[0]
                if checking_s1[1] == targets[0] and k == targets[1]:
                    intersect_point_1 = k
                    mu1 = 0
                if checking_s1[1] == targets[0] and k == targets[2]:
                    intersect_point_3 = k
                    mu3 = 0
                if k not in discover_dict_s1:
                    discover_dict_s1[k] = [checking_s1[1], child_distance]
                    tree_s1.append((child_distance, k))

                elif child_distance < discover_dict_s1[k][1]:
                    tree_s1.append((child_distance, k))
                    discover_dict_s1[k] = [checking_s1[1], child_distance]
                if k in discover_dict_s2 and continue1:
                # if k in scaned_dict_s2 and continue1:
                    current_path_1 = discover_dict_s1[k][1] + discover_dict_s2[k][1]
                    if mu1 > current_path_1: 
                        mu1 = current_path_1
                        intersect_point_1 = k
                if k in discover_dict_s3 and continue3:
                # if k in scaned_dict_s3 and continue3:
                    current_path_3 = discover_dict_s1[k][1] + discover_dict_s3[k][1]
                    if mu3 > current_path_3: 
                        mu3 = current_path_3
                        intersect_point_3 = k

            for k in graph[checking_s2[1]]:
                if k in scaned_dict_s2:
                    continue

                child_distance = graph.get_edge_weight(checking_s2[1], k) + checking_s2[0]
                if checking_s2[1] == targets[1] and k == targets[0]:
                    intersect_point_1 = k
                    mu1 = 0
                if checking_s2[1] == targets[1] and k == targets[2]:
                    intersect_point_2 = k
                    mu2 = 0
                if k not in discover_dict_s2:
                    discover_dict_s2[k] = [checking_s2[1], child_distance]
                    tree_s2.append((child_distance, k))

                elif child_distance < discover_dict_s2[k][1]:
                    tree_s2.append((child_distance, k))
                    discover_dict_s2[k] = [checking_s2[1], child_distance]
                if k in discover_dict_s1 and continue1:
                # if k in scaned_dict_s1 and continue1:
                    current_path_1 = discover_dict_s1[k][1] + discover_dict_s2[k][1]
                    if mu1 > current_path_1: 
                        mu1 = current_path_1
                        intersect_point_1 = k
                if k in discover_dict_s3 and continue2:
                # if k in scaned_dict_s3 and continue2:
                    current_path_2 = discover_dict_s2[k][1] + discover_dict_s3[k][1]
                    if mu2 > current_path_2: 
                        mu2 = current_path_2
                        intersect_point_2 = k
            
            for k in graph[checking_s3[1]]:
                if k in scaned_dict_s3:
                    continue

                child_distance = graph.get_edge_weight(checking_s3[1], k) + checking_s3[0]
                if checking_s3[1] == targets[2] and k == targets[0]:
                    intersect_point_3 = k
                    mu3 = 0
                if checking_s3[1] == targets[2] and k == targets[1]:
                    intersect_point_2 = k
                    mu2 = 0
                if k not in discover_dict_s3:
                    discover_dict_s3[k] = [checking_s3[1], child_distance]
                    tree_s3.append((child_distance, k))

                elif child_distance < discover_dict_s3[k][1]:
                    tree_s3.append((child_distance, k))
                    discover_dict_s3[k] = [checking_s3[1], child_distance]
                if k in discover_dict_s1 and continue3:
                # if k in scaned_dict_s1 and continue3:
                    current_path_3 = discover_dict_s1[k][1] + discover_dict_s3[k][1]
                    if mu3 > current_path_3: 
                        mu3 = current_path_3
                        intersect_point_3 = k
                if k in discover_dict_s2 and continue2:
                # if k in scaned_dict_s2 and continue2:
                    current_path_2 = discover_dict_s2[k][1] + discover_dict_s3[k][1]
                    if mu2 > current_path_2: 
                        mu2 = current_path_2
                        intersect_point_2 = k    

        heapq.heapify(path_list)
        path1 = heapq.heappop(path_list)
        path2 = heapq.heappop(path_list)

        if path1[1][0] == path2[1][0]:
            path1[1].reverse()
            p2 = path2[1][1:]
            path = path1[1] + p2

        elif path1[1][0] == path2[1][-1]:
            path2[1].pop()
            path = path2[1] + path1[1]

        elif path1[1][-1] == path2[1][0]:
            path1[1].pop()
            path = path1[1] + path2[1]
        
        else:
            path1[1].reverse()
            path2[1].pop()
            path = path2[1] + path1[1]
        return path
        
    raise NotImplementedError

def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    node_list = list(graph.nodes())
    length = len(node_list)
    if length < 4:
        return None
    elif length == 4:
        landmarks_list = []
        # s = node_list[0]
        for i in range(4):
            start = node_list[i]
            tree = PriorityQueue()
            discover_dict={}
            tree.append((0, start))
            discover_dict[start] = [start, 0]
            node_value = {}
            node_value[start] = 0
            ctn = True

            while tree.size()>0:
                checking = tree.pop()
                while checking[0] != discover_dict[checking[1]][1]:
                    if tree.size()>0:
                        checking = tree.pop()
                    else:
                        ctn = False
                        break
                if ctn == False:
                    break    
                node_value[checking[1]] = checking[0]

                for k in graph[checking[1]]:
                    child_distance = graph.get_edge_weight(checking[1], k) + checking[0]

                    if k not in discover_dict:
                        discover_dict[k] = [checking[1], child_distance]
                        tree.append((child_distance, k))

                    elif child_distance < discover_dict[k][1]:
                        tree.append((child_distance, k))
                        discover_dict[k] = [checking[1], child_distance]
                
            landmarks_list.append(node_value)
        return landmarks_list
    else:
        landmarks_list = []

        s_n = math.ceil(length/2) - 1
        s = node_list[s_n]
        # s = random.choice(node_list)
        ed = []
        for item in node_list:
            p1 = graph.nodes[item]['pos']
            p2 = graph.nodes[s]['pos']
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            ed.append((-d,item))
            
        heapq.heapify(ed)
        _, lm_1 = heapq.heappop(ed)

        start = lm_1
        tree = PriorityQueue()
        discover_dict={}
        tree.append((0, start))
        discover_dict[start] = [start, 0]
        node_value = {}
        node_value[start] = 0

        while tree.size()>0:
            checking = tree.pop()
            if checking[0] != discover_dict[checking[1]][1]:
                continue

            node_value[checking[1]] = checking[0]

            for k in graph[checking[1]]:
                child_distance = graph.get_edge_weight(checking[1], k) + checking[0]

                if k not in discover_dict:
                    discover_dict[k] = [checking[1], child_distance]
                    tree.append((child_distance, k))

                elif child_distance < discover_dict[k][1]:
                    tree.append((child_distance, k))
                    discover_dict[k] = [checking[1], child_distance]
        landmarks_list.append(node_value)
        
        ed = []
        for item in node_list:
            p1 = graph.nodes[item]['pos']
            p2 = graph.nodes[lm_1]['pos']
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            ed.append((-d,item))
            
        heapq.heapify(ed)
        _, lm_2 = heapq.heappop(ed)

        start = lm_2
        tree = PriorityQueue()
        discover_dict={}
        tree.append((0, start))
        discover_dict[start] = [start, 0]
        node_value = {}
        node_value[start] = 0
        ctn = True

        while tree.size()>0:
            checking = tree.pop()
            if checking[0] != discover_dict[checking[1]][1]:
                continue

            node_value[checking[1]] = checking[0]

            for k in graph[checking[1]]:
                child_distance = graph.get_edge_weight(checking[1], k) + checking[0]

                if k not in discover_dict:
                    discover_dict[k] = [checking[1], child_distance]
                    tree.append((child_distance, k))

                elif child_distance < discover_dict[k][1]:
                    tree.append((child_distance, k))
                    discover_dict[k] = [checking[1], child_distance]
        landmarks_list.append(node_value)

        ed = []
        for item in node_list:
            p1 = graph.nodes[item]['pos']
            p2 = graph.nodes[lm_1]['pos']
            p3 = graph.nodes[lm_2]['pos']
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1]) + math.hypot(p1[0] - p3[0], p1[1] - p3[1])
            ed.append((-d,item))
            
        heapq.heapify(ed)
        _, lm_3 = heapq.heappop(ed)
        if lm_3 == lm_1 or lm_3 == lm_2:
            _, lm_3 = heapq.heappop(ed)

        start = lm_3
        tree = PriorityQueue()
        discover_dict={}
        tree.append((0, start))
        discover_dict[start] = [start, 0]
        node_value = {}
        node_value[start] = 0
        ctn = True

        while tree.size()>0:
            checking = tree.pop()
            if checking[0] != discover_dict[checking[1]][1]:
                
                continue

            node_value[checking[1]] = checking[0]

            for k in graph[checking[1]]:
                child_distance = graph.get_edge_weight(checking[1], k) + checking[0]

                if k not in discover_dict:
                    discover_dict[k] = [checking[1], child_distance]
                    tree.append((child_distance, k))

                elif child_distance < discover_dict[k][1]:
                    tree.append((child_distance, k))
                    discover_dict[k] = [checking[1], child_distance]
        landmarks_list.append(node_value)

        ed = []
        for item in node_list:
            p1 = graph.nodes[item]['pos']
            p2 = graph.nodes[lm_3]['pos']
            d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            ed.append((-d,item))
            
        heapq.heapify(ed)
        node_value = {}

        while len(node_value) != len(landmarks_list[0]):
            _, lm_4 = heapq.heappop(ed)
            if lm_4 == lm_1 or lm_4 == lm_2 or lm_4 == lm_3:
                _, lm_4 = heapq.heappop(ed)

            start = lm_4
            tree = PriorityQueue()
            discover_dict={}
            tree.append((0, start))
            discover_dict[start] = [start, 0]
            node_value = {}
            node_value[start] = 0
        
            while tree.size()>0:
                checking = tree.pop()
                if checking[0] != discover_dict[checking[1]][1]:
                    continue

                node_value[checking[1]] = checking[0]

                for k in graph[checking[1]]:
                    child_distance = graph.get_edge_weight(checking[1], k) + checking[0]

                    if k not in discover_dict:
                        discover_dict[k] = [checking[1], child_distance]
                        tree.append((child_distance, k))

                    elif child_distance < discover_dict[k][1]:
                        tree.append((child_distance, k))
                        discover_dict[k] = [checking[1], child_distance]

        landmarks_list.append(node_value)

        return landmarks_list

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=compute_landmarks):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    # lm_list  = landmarks(graph)
    lm_list  = landmarks

    targets = set(goals)
    targets = list(targets)
    L = len(targets)
    d_g = []

    if L == 1:
        return []
    if L == 2:
        start = targets[0]
        goal =  targets[1]
        tree = PriorityQueue()
       
        discover_dict={}

        lm_score1 = 0
        lm_score2 = 0
        lm_score3 = 0
        lm_score4 = 0

        if goal in lm_list[0] and start in lm_list[0]:
            lm_score1 = abs(lm_list[0][start] - lm_list[0][goal])

        if goal in lm_list[1] and start in lm_list[1]:
            lm_score2 = abs(lm_list[1][start] - lm_list[1][goal])

        if goal in lm_list[2] and start in lm_list[2]:
            lm_score3 = abs(lm_list[2][start] - lm_list[2][goal])

        if goal in lm_list[3] and start in lm_list[3]:
            lm_score4 = abs(lm_list[3][start] - lm_list[3][goal])

        d_start = max(lm_score1, lm_score2, lm_score3, lm_score4)

        current_d = 0
        tree.append((d_start, start))
        discover_dict[start] = [start, d_start, current_d]
        path = []

        while tree.size()>0:
            lm_score1 = 0
            lm_score2 = 0
            lm_score3 = 0
            lm_score4 = 0

            checking = tree.pop()
            if checking[0] != discover_dict[checking[1]][1]:
                continue

            if checking[1] == goal:
                trace_back = checking[1]
                path.append(trace_back)
                while trace_back != start:
                    path.append(discover_dict[trace_back][0])
                    trace_back = discover_dict[trace_back][0]

                path.reverse()
                return path

            for m in graph[checking[1]]:
                current_d = graph.get_edge_weight(checking[1], m) + discover_dict[checking[1]][2]

                lm_score1 = abs(lm_list[0][m] - lm_list[0][goal])
                lm_score2 = abs(lm_list[1][m] - lm_list[1][goal])
                lm_score3 = abs(lm_list[2][m] - lm_list[2][goal])
                lm_score4 = abs(lm_list[3][m] - lm_list[3][goal])

                h_value = max(lm_score1, lm_score2, lm_score3, lm_score4)
                

                child_distance = current_d + h_value
                
                if m not in discover_dict:
                    discover_dict[m] = [checking[1], child_distance, current_d]
                    tree.append((child_distance, m))

                elif child_distance < discover_dict[m][1]:
                    tree.append((child_distance, m))
                    discover_dict[m] = [checking[1], child_distance, current_d]
    else:
        for k in range(L):
            if k == 2:
                d = heuristic(graph, targets[k], targets[0])
                d_g.append((d, targets[k], targets[0]))
            else:
                d = heuristic(graph, targets[k], targets[k+1])
                d_g.append((d, targets[k], targets[k+1]))
        heapq.heapify(d_g)

        # test 1
        s1 = heapq.heappop(d_g)
        s2 = heapq.heappop(d_g)
        s0 = set(s1[1:])&set(s2[1:])
        g0 = set(s1[1:])^set(s2[1:])
        start_list = list(s0)
        goal_list = list(g0)

        if len(lm_list) < 4:
            if L == 2:
                return [start_list[0], start_list[1]]
            if L == 3:
                glist = list(s1^s2)
                path = [glist[0]] + list(s1&s2) + [glist[1]]
                print(L, glist, path)
                return path

        # test 1
        path_list = []    
        start = start_list[0]
        goal = goal_list
        tree_1 = PriorityQueue()
        tree_2 = PriorityQueue()
        discover_dict={}

        lm_score1_1 = abs(lm_list[0][start] - lm_list[0][goal[0]])
        lm_score2_1 = abs(lm_list[1][start] - lm_list[1][goal[0]])
        lm_score3_1 = abs(lm_list[2][start] - lm_list[2][goal[0]])
        lm_score4_1 = abs(lm_list[3][start] - lm_list[3][goal[0]])

        lm_score1_2 = abs(lm_list[0][start] - lm_list[0][goal[1]])
        lm_score2_2 = abs(lm_list[1][start] - lm_list[1][goal[1]])
        lm_score3_2 = abs(lm_list[2][start] - lm_list[2][goal[1]])
        lm_score4_2 = abs(lm_list[3][start] - lm_list[3][goal[1]])

        d_start_1 = max(lm_score1_1, lm_score2_1, lm_score3_1, lm_score4_1)
        d_start_2 = max(lm_score1_2, lm_score2_2, lm_score3_2, lm_score4_2)

        current_d = 0
        tree_1.append((d_start_1, start))
        tree_2.append((d_start_2, start))
        discover_dict[start] = [start, start, d_start_1, d_start_2, current_d]
        continue_1 = True
        continue_2 = True
        checked_node = {}
        pop_1 = True
        pop_2 = True
        path1 = []
        path2 = []

        while tree_1.size()>0 and tree_2.size()>0:

            if not continue_1 and not continue_2:
                break
            if not pop_1 and continue_1:
                next_pop_1 = tree_1.pop()
                if next_pop_1[0] < checking_1[0]:
                    tree_1.append(checking_1)
                    checking_1 = next_pop_1
                else:
                    tree_1.append(next_pop_1)

            if not pop_2 and continue_2:
                next_pop_2 = tree_2.pop()
                if next_pop_2[0] < checking_2[0]:
                    tree_2.append(checking_2)
                    checking_2 = next_pop_2
                else:
                    tree_2.append(next_pop_2)

            if continue_1 and pop_1:
                checking_1 = tree_1.pop()
 
                while (checking_1[0] != discover_dict[checking_1[1]][2]) or (checking_1[1] in checked_node):
                    checking_1 = tree_1.pop()

            if continue_2 and pop_2:
                checking_2 = tree_2.pop()
                while (checking_2[0] != discover_dict[checking_2[1]][3]) or (checking_2[1] in checked_node):
                    checking_2 = tree_2.pop()

            if checking_1[0] <= checking_2[0]:
                if continue_1:
                    checking_node = checking_1
                    pop_2 = False
                    pop_1 = True
                    if checking_node[1] == checking_2[1]:
                        pop_2 = True
                else:
                    checking_node = checking_2
                    pop_2 = True
            else:
                if continue_2:
                    checking_node = checking_2
                    pop_1 = False
                    pop_2 = True
                    if checking_node[1] == checking_1[1]:
                        pop_1 = True
                else:
                    checking_node = checking_1
                    pop_1 = True

            checked_node[checking_node[1]] = 0

            if checking_node[1] == goal[0] and continue_1:
                trace_back = checking_node[1]
                path1.append(trace_back)
                while trace_back != start:
                    path1.append(discover_dict[trace_back][0])
                    trace_back = discover_dict[trace_back][0]

                path1.reverse()
  
                if goal[1] in path1:
                    return path1

                path_list.append((discover_dict[checking_node[1]][4], path1))
                continue_1 = False
                if checking_2[1] in checked_node:
                    pop_2 = True

            if checking_node[1] == goal[1] and continue_2:
                trace_back = checking_node[1]
                path2.append(trace_back)
                while trace_back != start:
                    path2.append(discover_dict[trace_back][1])
                    trace_back = discover_dict[trace_back][1]

                path2.reverse()

                if goal[0] in path2:
                    return path2

                path_list.append((discover_dict[checking_node[1]][4], path2))
                continue_2 = False
                if checking_1[1] in checked_node:
                    pop_1 = True

            for m in graph[checking_node[1]]:
                if m in checked_node:
                    continue

                current_d = graph.get_edge_weight(checking_node[1], m) + discover_dict[checking_node[1]][4]
                if continue_1:
                    lm_score1_1 = abs(lm_list[0][m] - lm_list[0][goal[0]])
                    lm_score2_1 = abs(lm_list[1][m] - lm_list[1][goal[0]])               
                    lm_score3_1 = abs(lm_list[2][m] - lm_list[2][goal[0]])               
                    lm_score4_1 = abs(lm_list[3][m] - lm_list[3][goal[0]])

                    h_value_1 = max(lm_score1_1, lm_score2_1, lm_score3_1, lm_score4_1)
                    child_distance_1 = current_d + h_value_1
   
                if continue_2:
                    lm_score1_2 = abs(lm_list[0][m] - lm_list[0][goal[1]])
                    lm_score2_2 = abs(lm_list[1][m] - lm_list[1][goal[1]])               
                    lm_score3_2 = abs(lm_list[2][m] - lm_list[2][goal[1]])               
                    lm_score4_2 = abs(lm_list[3][m] - lm_list[3][goal[1]])

                    h_value_2 = max(lm_score1_2, lm_score2_2, lm_score3_2, lm_score4_2)
                    child_distance_2 = current_d + h_value_2

                if m not in discover_dict:
                    discover_dict[m] = [checking_node[1], checking_node[1], child_distance_1, child_distance_2, current_d]
                    tree_1.append((child_distance_1, m))
                    tree_2.append((child_distance_2, m))
                
                cd_1 = discover_dict[m][2]
                cd_2 = discover_dict[m][3]

                if continue_1 and child_distance_1 < cd_1:
                    tree_1.append((child_distance_1, m))
                    discover_dict[m] = [checking_node[1], checking_node[1], child_distance_1, child_distance_2, current_d]

                if continue_2 and child_distance_2 < cd_2:
                    tree_2.append((child_distance_2, m))
                    discover_dict[m] = [checking_node[1], checking_node[1], child_distance_1, child_distance_2, current_d]

        path_a = path_list[0][1].copy()
        path_b = path_list[1][1].copy()
        intersect_points = set(path_a).intersection(set(path_b))
        path_a.reverse()
        for p in path_a:
            if p in intersect_points:
                cross_point = p
        
        s_c = discover_dict[cross_point][-1]
        # print(path_a, path_b, cross_point, s_c)
        c_g1 = path_list[0][0] - s_c
        c_g2 = path_list[1][0] - s_c
        if c_g1 <= c_g2:
            g1 = True
        else:
            g1 = False

        c_g = min(c_g1, c_g2)
        if c_g <= s_c:
            ub = c_g1 + c_g2
        else:
            ub = s_c + max(c_g1, c_g2)
        # print(ub)
        start = goal_list[0]
        goal =  goal_list[1]
        tree = PriorityQueue()
       
        discover_dict={}

        lm_score1 = 0
        lm_score2 = 0
        lm_score3 = 0
        lm_score4 = 0

        lm_score1 = abs(lm_list[0][start] - lm_list[0][goal])
        lm_score2 = abs(lm_list[1][start] - lm_list[1][goal])
        lm_score3 = abs(lm_list[2][start] - lm_list[2][goal])
        lm_score4 = abs(lm_list[3][start] - lm_list[3][goal])

        d_start = max(lm_score1, lm_score2, lm_score3, lm_score4)

        current_d = 0
        tree.append((d_start, start))
        discover_dict[start] = [start, d_start, current_d]
        path = []

        while tree.size()>0:
            lm_score1 = 0
            lm_score2 = 0
            lm_score3 = 0
            lm_score4 = 0

            checking = tree.pop()
            if checking[0] != discover_dict[checking[1]][1]:
                continue

            if discover_dict[checking[1]][1] >= ub:
                # print(checking[0])
                if s_c <= c_g:
                    break
                else:
                    if g1:
                        path = path_list[0][1].copy()
                        path.pop()
                        path_list[0][1].reverse()
                        for point in path_list[0][1]:
                            if point != cross_point:
                                path.append(point)

                        path_list[1][1].reverse()
                        for point in path_list[1][1]:
                            if point != cross_point:
                                path_list[1][1].pop()
                        path_list[1][1].reverse()
                        path = path + path_list[1][1]
                        return path        
                    else:
                        path = path_list[1][1].copy()
                        path.pop()
                        path_list[1][1].reverse()
                        for point in path_list[0][1]:
                            if point != cross_point:
                                path.append(point)

                        path_list[0][1].reverse()
                        for point in path_list[0][1]:
                            if point != cross_point:
                                path_list[0][1].pop()
                        path_list[0][1].reverse()
                        path = path + path_list[0][1]
                        return path

            if checking[1] == goal:
                trace_back = checking[1]
                path.append(trace_back)
                while trace_back != start:
                    path.append(discover_dict[trace_back][0])
                    trace_back = discover_dict[trace_back][0]

                path.reverse()
                path_list.append((discover_dict[checking[1]][2], path))
                break

            for m in graph[checking[1]]:
                current_d = graph.get_edge_weight(checking[1], m) + discover_dict[checking[1]][2]
                
                lm_score1 = abs(lm_list[0][m] - lm_list[0][goal])
                lm_score2 = abs(lm_list[1][m] - lm_list[1][goal])
                lm_score3 = abs(lm_list[2][m] - lm_list[2][goal])
                lm_score4 = abs(lm_list[3][m] - lm_list[3][goal])

                h_value = max(lm_score1, lm_score2, lm_score3, lm_score4) 

                child_distance = current_d + h_value
                
                if m not in discover_dict:
                    discover_dict[m] = [checking[1], child_distance, current_d]
                    tree.append((child_distance, m))

                elif child_distance < discover_dict[m][1]:
                    tree.append((child_distance, m))
                    discover_dict[m] = [checking[1], child_distance, current_d]

        heapq.heapify(path_list)
        _, p1 = heapq.heappop(path_list)
        _, p2 = heapq.heappop(path_list)   
        if p1[0] == p2[0]:
            p1.reverse()
            path = p1 + p2[1:]

        elif p1[0] == p2[-1]:
            p2.pop()
            path = p2 + p1

        elif p1[-1] == p2[0]:
            p1.pop()
            path = p1 + p2

        else:
            p1.reverse()
            p2.pop()
            path = p2 + p1
        return path 


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return 'Wenda Xu'
    # raise NotImplementedError


# def compute_landmarks(graph):
    # """
    # Feel free to implement this method for computing landmarks. We will call
    # tridirectional_upgraded() with the object returned from this function.

    # Args:
    #     graph (ExplorableGraph): Undirected graph to search.

    # Returns:
    # List with not more than 4 computed landmarks. 
    # """

    # return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
