from collections import defaultdict

from pymoo.core.problem import Problem
import numpy as np
from utils import *
from copy import copy


class RCPSP_RandomKeyRepresentation(Problem):
    def __init__(self, graph: dict, times_dict: dict, r_cap_dict={}, r_cons_dict={}, r_count=None, act_pre=None):
        self.graph = graph
        self.all_tasks = khan(graph.copy())
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
            n_constr=len(self.r_cap_dict.keys())*sum(times_dict.values()),
            xl=0,
            xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        indvs = [solution_from_random_key(indiv, self.graph.copy()) for indiv in x[:]]
        #https://github.com/bantosik/py-rcpsp/blob/a9c180f8425a60af9cb18971378b89d7843aea6f/SingleModeClasses.py#L1
        #criando dicionario de tempos de uso
        self.times_dict[0] = 0
        total_time_all_activit = sum(value for key, value in (self.times_dict.items()))

        # indvs_after_sgs = []
        makespans = []
        number_of_resources = len(self.r_cap_dict.keys())
        restrictions = [[0]*number_of_resources*total_time_all_activit]*len(indvs)
        for count, solution in enumerate(indvs[:]):
            # solution, resource_usages_in_time = serialSGS(ind, total_time_all_activit, self.r_count, self.r_cons_dict , self.r_cap_dict, self.times_dict, self.act_pre)
            resource_usages = {resource: np.array([0]*total_time_all_activit) for resource in self.r_cap_dict.keys()}
            start_times = [0]*len(solution)
            for activity in solution:
                self.update_resource_usages_and_start_times(start_times, activity, resource_usages, solution)
            mkspan = max(start_times) + self.times_dict[self.all_tasks[np.argmax(start_times)]]
            makespans.append(float(mkspan))
            for i, resource in enumerate(resource_usages.keys()):
                usage = resource_usages[resource]
                capacity = self.r_cap_dict[resource]
                # each resource has its range of restriction indices corresponding to their usage at each time step
                begin_index_resource = i * total_time_all_activit
                end_index_resource = (i + 1) * total_time_all_activit
                restrictions[count][begin_index_resource: end_index_resource] = usage - capacity
            assert all([solution.index(key) > solution.index(task) for key, value in self.graph.items() for task in value])
            assert all([start_times[self.all_tasks.index(key)] >= start_times[self.all_tasks.index(task)] + self.times_dict[task] for key, value in self.graph.items() for task in value])
        out["G"] = np.array(restrictions)
        infeasibility_value = np.sum(out["G"]*np.where(out["G"] <= 0, 0, 1), axis=1)

        # print(start_times)

        out["F"] = np.where(infeasibility_value <= 0, np.array(makespans), np.array(makespans) + 100*infeasibility_value)

    def update_resource_usages_and_start_times(self, start_times, activity, resource_usages, solution):
        if self.graph[activity] != []:
            activity_start_time = max(
                [start_times[self.all_tasks.index(other)] + self.times_dict[other] for other in self.graph[activity]]
            )
        else:
            activity_start_time = 0
        activity_ending_time = activity_start_time + self.times_dict[activity]
        feasible = self.check_feasible(activity, activity_start_time, resource_usages)
        while not feasible:
            # print(activity_start_time)
            ending_time = lambda task: start_times[self.all_tasks.index(task)] + self.times_dict[task]
            task_uses_resource = lambda task, resource: (defaultdict(int, self.r_cons_dict)[task, resource] > 0)
            start_time_defined = lambda task: solution.index(task) < solution.index(activity)
            started_b4_cur_task_ends = lambda task: start_time_defined(task) and start_times[self.all_tasks.index(task)] <= activity_ending_time
            doesnt_halt_b4_cur_task_starts = lambda task: ending_time(task) > activity_start_time
            is_on_the_right_time_window = lambda task: started_b4_cur_task_ends(task) and doesnt_halt_b4_cur_task_starts(task)
            common_resources = lambda task1, task2: [resource for resource in resource_usages.keys()
                                                     if task_uses_resource(task1, resource) and task_uses_resource(task2, resource)]
            ending_times_concurrent_tasks = [
                ending_time(other_task) for other_task in self.all_tasks
                if is_on_the_right_time_window(other_task) and len(common_resources(other_task, activity)) > 0
            ]
            activity_start_time = min(ending_times_concurrent_tasks)
            activity_ending_time = activity_start_time + self.times_dict[activity]
            feasible = self.check_feasible(activity, activity_start_time, resource_usages)
        start_times[self.all_tasks.index(activity)] = activity_start_time
        for resource in resource_usages.keys():
            if (activity, resource) in self.r_cons_dict:
                curr_activity_resource_usage = self.r_cons_dict[(activity, resource)]
            else:
                curr_activity_resource_usage = 0
            resource_usages[resource][activity_start_time:activity_ending_time] += curr_activity_resource_usage

    def check_feasible(self, activity, activity_start_time, resource_usages):
        activity_ending_time = activity_start_time + self.times_dict[activity]
        feasible = True
        for resource in resource_usages.keys():
            if (activity, resource) in self.r_cons_dict.keys() and self.r_cons_dict[(activity, resource)] > 0:
                already_used = resource_usages[resource][activity_start_time:activity_ending_time]
                curr_activity_resource_usage = self.r_cons_dict[(activity, resource)]
                capacity = self.r_cap_dict[resource]
                feasible = feasible and np.all(already_used + curr_activity_resource_usage <= capacity)
        return feasible


class RCPSP(Problem):
    def __init__(self, graph: dict, times_dict: dict, r_cap_dict={}, r_cons_dict={}):
        self.graph = graph
        self.all_tasks = list(set([task for k, v in graph.items() for task in [k] + v]))
        self.times_dict = times_dict
        self.r_cap_dict = r_cap_dict
        self.r_cons_dict = r_cons_dict
        self.time_step_size = min([t for t in times_dict.values() if t > 0])
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
        infeasibility_value = np.sum(restrictions*np.where(restrictions<=0,0,1), axis=1)
        out["F"] = np.where(infeasibility_value <= 0, ending_time_of_all_tasks, ending_time_of_all_tasks + infeasibility_value)



class RCPSP_RKR_debug(Problem):
    def __init__(self, graph: dict, times_dict: dict, r_cap_dict={}, r_cons_dict={}, r_count=None, act_pre=None):
        self.graph = graph
        self.all_tasks = khan(graph.copy())
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
            n_constr=len(self.r_cap_dict.keys())*sum(times_dict.values()),
            xl=0,
            xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        indvs = another_random_key_decoder(x, self.all_tasks)
        #https://github.com/bantosik/py-rcpsp/blob/a9c180f8425a60af9cb18971378b89d7843aea6f/SingleModeClasses.py#L1
        #criando dicionario de tempos de uso
        self.times_dict[0] = 0
        total_time_all_activit = sum(value for key, value in (self.times_dict.items()))

        indvs_after_sgs = []
        makespans = []
        number_of_resources = len(self.r_cap_dict.keys())
        restrictions = [[0]*number_of_resources*total_time_all_activit]*len(indvs)
        for count, ind in enumerate(indvs[:]):
            resource_usages = {resource: np.array([0]*total_time_all_activit) for resource in self.r_cap_dict.keys()}
            solution, resource_usages = serialSGS(ind, total_time_all_activit, self.r_count, self.r_cons_dict , self.r_cap_dict, self.times_dict, self.act_pre, self.graph, resource_usages)
            mkspan = compute_makespan(solution, self.times_dict)
            indvs_after_sgs.append(solution)
            makespans.append(float(mkspan))
            #FIM Serial SGS para todos os individuos
            for i, resource in enumerate(resource_usages.keys()):
                usage = resource_usages[resource]
                capacity = self.r_cap_dict[resource]
                # each resource has its range of restriction indices corresponding to their usage at each time step
                begin_index_resource = i * total_time_all_activit
                end_index_resource = (i + 1) * total_time_all_activit
                restrictions[count][begin_index_resource: end_index_resource] = usage - capacity
        # print("oi")
        # print(min(makespans))
        # print(indvs_after_sgs[makespans.index(min(makespans))])
        # print(resource_usages_in_time)
        # time.sleep(40)
        # print(restrictions)


        out["G"] = np.array(restrictions)
        infeasibility_value = np.sum(out["G"]*np.where(out["G"] <= 0, 0, 1), axis=1)
        out["F"] = np.where(infeasibility_value <= 0, np.array(makespans), np.array(makespans) + 100*infeasibility_value)