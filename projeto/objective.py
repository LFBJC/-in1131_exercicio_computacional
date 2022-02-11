from pymoo.core.problem import Problem
import numpy as np


class MRCPSP(Problem):
    def __init__(self, graph: dict, times_dict):
        self.graph = graph
        self.all_nodes = list(set([task for k, v in graph.items() for task in [k] + v]))
        self.times_dict = times_dict
        super().__init__(
            n_var=len(self.all_nodes),
            n_obj=1,
            n_constr=sum([len(v) for v in graph.values()])+1,
            xl=0,
            xu=sum(list(times_dict.values()))
        )

    def _evaluate(self, x, out, *args, **kwargs):
        max_start_times = np.max(x, axis=1)
        last_tasks_by_indiv = [self.all_nodes[max_time_index] for max_time_index in np.argmax(x, axis=1)]
        durations_of_last_tasks = np.array([self.times_dict[last_task] for last_task in last_tasks_by_indiv])
        # adjusting the format in order to get a column vector that may be summed up with the initial times of the last
        # tasks of each individual
        durations_of_last_tasks = np.transpose(durations_of_last_tasks)
        out["F"] = max_start_times + durations_of_last_tasks
        restrictions = np.array([])
        for node in self.all_nodes:
            if node in self.graph.keys():
                for requirement in self.graph[node]:
                    node_start_times = x[:, self.all_nodes.index(node)]
                    requirement_start_times = x[:, self.all_nodes.index(requirement)]
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
        out["G"] = restrictions

