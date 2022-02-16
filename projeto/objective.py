from pymoo.core.problem import Problem
import numpy as np
from utils import *
from copy import copy

class RCPSP_RandomKeyRepresentation(Problem):
    def __init__(self, graph: dict, times_dict: dict, r_cap_dict={}, r_cons_dict={}, r_count=None, act_pre=None):
        self.graph = graph
        self.all_tasks, self.no_outdegree_vertices = khan(graph)
        self.times_dict = times_dict
        self.r_cap_dict = r_cap_dict
        self.r_cons_dict = r_cons_dict
        self.time_step_size = min(times_dict.values())
        self.r_count = r_count
        self.act_pre = act_pre
        self.resources_to_constraint_vectors = {k: np.zeros_like(self.all_tasks) for k in self.r_cap_dict.keys()}
        for resource in self.resources_to_constraint_vectors.keys():
            task_indices = [self.all_tasks.index(task_resource_pair[0]) for task_resource_pair in self.r_cons_dict.keys() if task_resource_pair[1] == resource]
            for task_index in task_indices:
                self.resources_to_constraint_vectors[resource][task_index] = self.r_cons_dict[(self.all_tasks[task_index], resource)]
        super().__init__(
            n_var=len(self.all_tasks),
            n_obj=1,
            n_constr=sum([len(v) for v in graph.values()])+len(self.r_cap_dict.keys())*sum(times_dict.values()),
            xl=0,
            xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        indvs = another_random_key_decoder(x, self.all_tasks)
        #https://github.com/bantosik/py-rcpsp/blob/a9c180f8425a60af9cb18971378b89d7843aea6f/SingleModeClasses.py#L1
        #criando dicionario de tempos de uso
        
        total_time_all_activit = sum(value for key, value in (self.times_dict.items()))

        indvs_after_sgs = []
        makespans = []

        #Serial SGS para todos os individuos

        for ind in indvs[:]:
            solution = []
            resource_usages_in_time = {}
            time_points = [0]

            for sec in range(total_time_all_activit+1):
                resource_usages_in_time[sec] = {}
                for key, value in self.r_cap_dict.items():
                    key_aux = {key: (value,0)} #tupla capacidade/em uso
                    resource_usages_in_time[sec].update(key_aux)
            
            solution.append({0: 0})  #inicializando dummy
            for activity in ind:
                activity = int(activity)
                last_time = time_points[-1]
                start_time = 0
                for time_unit in reversed(time_points):
                    actual_resource_usage = copy(resource_usages_in_time[time_unit])
                    actual_resource_usage = add_resource_usage(actual_resource_usage, self.r_count, self.r_cons_dict, activity)
                    if is_resource_usage_greater_than_supply(self.r_count, actual_resource_usage) or activity_in_conflict_in_precedence(self.act_pre, activity, time_unit, self.times_dict, solution):
                            start_time = last_time
                            break
                    else:
                        last_time = time_unit
                tuple_start_time = {activity: start_time}
                solution.append(tuple_start_time)
                time_points = insert_value_to_ordered_list(time_points, start_time)
                time_points = insert_value_to_ordered_list(time_points, start_time + self.times_dict[activity])   
                resource_usages_in_time = update_resource_usages_in_time(resource_usages_in_time, activity, start_time, self.times_dict, self.r_count, self.r_cons_dict)
            indvs_after_sgs.append(solution)
            sol_sorted_by_values = sorted(solution, key=lambda d: list(d.values())) 
            mkspan = list(sol_sorted_by_values[-1].items())[0][1] + self.times_dict[list(sol_sorted_by_values[-1].items())[0][0]]
            makespans.append(mkspan)
        
        best_makespan = min(makespans)
        best_sol_idx = makespans.index(best_makespan)

        out["F"] = min(makespans)

        # resource constraints
        for resource, capacity in self.r_cap_dict.items():
            resource_constraint_matrix = np.repeat(
                self.resources_to_constraint_vectors[resource][np.newaxis, :],
                x.shape[0],
                axis=0
            )

            for time_interval_start in np.arange(0, sum(self.times_dict.values()), self.time_step_size):
                tasks_at_this_moment = np.logical_and(x >= time_interval_start, x < time_interval_start+self.time_step_size).astype(np.int32)*resource_constraint_matrix
                if restrictions.shape == (0,):
                    restrictions = np.sum(tasks_at_this_moment, axis=1)
                else:
                    restrictions = np.column_stack([
                        restrictions,
                        np.sum(tasks_at_this_moment, axis=1)
                    ])
        print(restrictions)
        out["G"] = restrictions


class RCPSP(Problem):
    def __init__(self, graph: dict, times_dict: dict, r_cap_dict={}, r_cons_dict={}):
        self.graph = graph
        self.all_tasks = list(set([task for k, v in graph.items() for task in [k] + v]))
        self.times_dict = times_dict
        self.r_cap_dict = r_cap_dict
        self.r_cons_dict = r_cons_dict
        self.time_step_size = min(times_dict.values())
        self.resources_to_constraint_vectors = {k: np.zeros_like(self.all_tasks) for k in self.r_cap_dict.keys()}
        for resource in self.resources_to_constraint_vectors.keys():
            task_indices = [self.all_tasks.index(task_resource_pair[0]) for task_resource_pair in self.r_cons_dict.keys() if task_resource_pair[1] == resource]
            for task_index in task_indices:
                self.resources_to_constraint_vectors[resource][task_index] = self.r_cons_dict[(self.all_tasks[task_index], resource)]
        super().__init__(
            n_var=len(self.all_tasks),
            n_obj=1,
            n_constr=sum([len(v) for v in graph.values()])+len(self.r_cap_dict.keys())*sum(times_dict.values()),
            xl=0,
            xu=sum(list(times_dict.values()))
        )

    def _evaluate(self, x, out, *args, **kwargs):
        max_start_times = np.max(x, axis=1)
        last_tasks_by_indiv = [self.all_tasks[max_time_index] for max_time_index in np.argmax(x, axis=1)]
        durations_of_last_tasks = np.array([self.times_dict[last_task] for last_task in last_tasks_by_indiv])
        # adjusting the format in order to get a column vector that may be summed up with the initial times of the last
        # tasks of each individual
        durations_of_last_tasks = np.transpose(durations_of_last_tasks)
        ending_time_of_all_tasks = max_start_times + durations_of_last_tasks
        out["F"] = ending_time_of_all_tasks
        # precedence constraints
        restrictions = np.array([])
        for node in self.all_tasks:
            if node in self.graph.keys():
                for requirement in self.graph[node]:
                    node_start_times = x[:, self.all_tasks.index(node)]
                    requirement_start_times = x[:, self.all_tasks.index(requirement)]
                    requirement_duration_vector = np.full(shape=x.shape[0], fill_value=self.times_dict[requirement])
                    if restrictions.shape == (0,):
                        restrictions = requirement_start_times+requirement_duration_vector-node_start_times
                    else:
                        restrictions = np.column_stack([
                            restrictions,
                            requirement_start_times+requirement_duration_vector-node_start_times
                        ])
            #  requirement_start_times+self.times_dict[requirement] -node_start_times <= 0
            # -> requirement_start_times+self.times_dict[requirement] <= node_start_times
        # resource constraints
        for resource, capacity in self.r_cap_dict.items():
            resource_constraint_matrix = np.repeat(
                self.resources_to_constraint_vectors[resource][np.newaxis, :],
                x.shape[0],
                axis=0
            )

            for time_interval_start in np.arange(0, sum(self.times_dict.values()), self.time_step_size):
                tasks_at_this_moment = np.logical_and(x >= time_interval_start, x < time_interval_start+self.time_step_size).astype(np.int32)*resource_constraint_matrix
                if restrictions.shape == (0,):
                    restrictions = np.sum(tasks_at_this_moment, axis=1)
                else:
                    restrictions = np.column_stack([
                        restrictions,
                        np.sum(tasks_at_this_moment, axis=1)
                    ])
        out["G"] = restrictions

