from pymoo.core.problem import Problem
import numpy as np
from utils import khan


class RCPSP_RandomKeyRepresentation(Problem):
    def __init__(self, graph: dict, times_dict: dict, r_cap_dict={}, r_cons_dict={}):
        self.graph = graph
        self.all_tasks = khan(graph)
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
            xu=1
        )

    def _evaluate(self, x, out, *args, **kwargs):
        indvs = x
        #RandomKey convert
        for arr in range(len(indvs)):
            ordenate_x = np.sort(indvs[arr])
            for i in range(len(ordenate_x)):
                idx, = np.where(indvs[arr] == ordenate_x[i])
                indvs[arr][idx[0]] = i+1
        #https://github.com/bantosik/py-rcpsp/blob/a9c180f8425a60af9cb18971378b89d7843aea6f/SingleModeClasses.py#L1
        out["F"] = 1

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

