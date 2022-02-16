import json
from objective import MRCPSP


def problem_from_json(file_name):
    with open(file_name) as json_file:
        problem_dict = json.load(json_file)
    act_pre = problem_dict['act_pre']
    graph = {(k): (act_pre[act_pre.index(k)] if k in act_pre else []) for k in range(1, problem_dict['act_count']+1)}
    times_dict = {d['index']: d['value'] for d in problem_dict['act_proc']}
    r_cap_dict = {d['index']: d['value'] for d in problem_dict['r_cap']}
    r_cons_dict = {tuple(d['index']): d['value'] for d in problem_dict['r_cons']}
    return graph, times_dict, r_cap_dict, r_cons_dict
    #return MRCPSP(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)


def random_key_decoder(x, reference_list):
    sorted_x = list(sorted(x))
    return [reference_list[sorted_x.index(e)] for e in x]


def khan(graph):
    ret = []
    no_incoming_edge = [key for key in graph.keys() if graph[key] == []]
    while no_incoming_edge != []:
        current_node, no_incoming_edge = no_incoming_edge, graph[no_incoming_edge]
        ret.append(current_node)
        for i, component in enumerate(graph.items()):
            if current_node in component[1]:
                graph[i][1] = [node for node in component[1] if node != current_node]
                if graph[i][1] == []:
                    no_incoming_edge.append(component[0])
    edges = [(key, x) for key, value in graph.items() if value != [] for x in value]
    if edges != []:
        raise ValueError("The graph has cycles")
    else:
        return ret
