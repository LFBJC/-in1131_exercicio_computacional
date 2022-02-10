from pymoo.core.problem import Problem
import numpy as np


class MRCPSP(Problem):
    def __init__(self, graph: dict, times_dict):
        self.graph = graph
        self.all_nodes = list(set([task for k, v in graph.items() for task in [k] + v]))
        self.times_dict = times_dict
        super().__init__(
            n_var=len(self.all_nodes)+1,
            n_obj=1,
            n_constr=sum([len(v) for v in graph.values()])+1,
            xl=0,
            xu=sum(list(times_dict.values()))
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[:, -1]
        restrictions = -x[:, -1] + np.max(x[:, :-1], axis=1)
        for node in self.all_nodes:
            if node in self.graph.keys():
                for dependent_task in self.graph[node]:
                    node_start_times = x[:, self.all_nodes.index(node)]
                    dependent_task_start_times = x[:, self.all_nodes.index(dependent_task)]
                    restrictions = np.column_stack([
                        restrictions,
                        node_start_times-dependent_task_start_times+self.times_dict[node]
                    ])
        out["G"] = restrictions

