import math
import random
from collections import deque

import config
from sprites import Goal, Spaceship
from state import State

import sys

sys.setrecursionlimit(50000)


class Algorithm:
    def get_path(self, state):
        pass


class ExampleAlgorithm(Algorithm):
    def get_path(self, state):
        path = []
        while not state.is_goal_state():
            possible_actions = state.get_legal_actions()
            action = possible_actions[random.randint(0, len(possible_actions) - 1)]
            path.append(action)
            state = state.generate_successor_state(action)
        return path


# DFS algorithm
class Blue(Algorithm):
    def get_path(self, state):
        path = []
        visited = set()
        if self.dfs(state, path, visited):
            return path
        else:
            return []

    def dfs(self, state, path, visited):
        if state.is_goal_state():
            return True

        visited.add(state.get_state(Spaceship.kind()))

        possible_actions = state.get_legal_actions()

        for action in possible_actions:
            next_state = state.generate_successor_state(action)
            if next_state.get_state(Spaceship.kind()) in visited:
                continue

            path.append(action)

            if self.dfs(next_state, path, visited):
                return True

            path.pop()

        return False


# BFS
class Red(Algorithm):
    def get_path(self, state):
        queue = deque([(state, [])])
        visited = set()

        while queue:
            current_state, path = queue.popleft()

            if current_state.is_goal_state():
                return path

            if current_state.get_state(Spaceship.kind()) in visited:
                continue

            visited.add(current_state.get_state(Spaceship.kind()))
            possible_actions = current_state.get_legal_actions()

            for action in possible_actions:
                next_state = current_state.generate_successor_state(action)
                new_path = path + [action]
                queue.append((next_state, new_path))
        return []


# B&B
class Black(Algorithm):
    def get_path(self, state):
        queue = [(0, state, [])]  # (cumulative_cost, current_state, path)
        visited = {}
        best_cost = math.inf
        best_path = []

        while queue:
            queue.sort(key=lambda x: x[0])  # Sort the queue based on cumulative cost
            cumulative_cost, current_state, path = queue.pop(0)

            # Check if the current path cost already exceeds the known best cost
            if cumulative_cost >= best_cost:
                continue

            if current_state.is_goal_state():
                if cumulative_cost < best_cost:
                    best_cost = cumulative_cost
                    best_path = path
                continue

            state_repr = current_state.get_state(Spaceship.kind())

            # Check if we have visited this state with a lower cost
            if state_repr in visited and visited[state_repr] <= cumulative_cost:
                continue

            visited[state_repr] = cumulative_cost
            possible_actions = current_state.get_legal_actions()

            for action in possible_actions:
                next_state = current_state.generate_successor_state(action)
                action_cost = State.get_action_cost(action)
                new_cumulative_cost = cumulative_cost + action_cost
                new_path = path + [action]

                if new_cumulative_cost < best_cost:
                    queue.append((new_cumulative_cost, next_state, new_path))

        return best_path if best_path else []


# A*
class White(Algorithm):
    def get_path(self, state):
        queue = [(0, 0, state, [])]  # (total_estimated_cost, cumulative_cost, current_state, path)
        visited = {}

        while queue:
            queue.sort(key=lambda x: x[0])  # Sort the queue based on total estimated cost
            total_estimated_cost, cumulative_cost, current_state, path = queue.pop(0)

            # Check if the current state is a goal
            if current_state.is_goal_state():
                return path

            state_repr = current_state.get_state(Spaceship.kind())

            # Check if we have visited this state with a lower or equal cost
            if state_repr in visited and visited[state_repr] <= cumulative_cost:
                continue

            visited[state_repr] = cumulative_cost
            possible_actions = current_state.get_legal_actions()

            for action in possible_actions:
                next_state = current_state.generate_successor_state(action)
                action_cost = State.get_action_cost(action)
                new_cumulative_cost = cumulative_cost + action_cost

                heuristic = self.manhattan_distance(next_state)
                total_cost = new_cumulative_cost + heuristic
                new_path = path + [action]

                queue.append((total_cost, new_cumulative_cost, next_state, new_path))

        return []

    @staticmethod
    def manhattan_distance(state):
        spaceships_position = state.get_state(Spaceship.kind())
        goals_position = state.get_state(Goal.kind())
        distance = 0
        total_distance = 0

        for i in range(config.M * config.N):
            closest_goal_distance = math.inf
            if spaceships_position & (1 << i):
                spaceship_x, spaceship_y = i // config.N, i % config.N
                for j in range(config.M * config.N):
                    if goals_position & (1 << i):
                        goal_x, goal_y = i // config.N, i % config.N
                        distance += abs(spaceship_x - goal_x) + abs(spaceship_y - goal_y)
                        closest_goal_distance = min(closest_goal_distance, distance)
            total_distance += closest_goal_distance

        return distance

    @staticmethod
    def euclidean_distance(state):
        spaceships_position = state.get_state(Spaceship.kind())
        goals_position = state.get_state(Goal.kind())
        distance = 0
        total_distance = 0

        for i in range(config.M * config.N):
            closest_goal_distance = float('inf')
            if spaceships_position & (1 << i):
                spaceship_x, spaceship_y = i // config.N, i % config.N
                for j in range(config.M * config.N):
                    if goals_position & (1 << j):
                        goal_x, goal_y = j // config.N, j % config.N
                        distance += math.sqrt((spaceship_x - goal_x) ** 2 + (spaceship_y - goal_y) ** 2)
                        closest_goal_distance = min(closest_goal_distance, distance)
            total_distance += closest_goal_distance

        return total_distance