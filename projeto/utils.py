import json

def problem_from_json(file_name):
    with open(file_name) as json_file:
        problem_dict = json.load(json_file)
    act_pre = problem_dict['act_pre']
    #graph = {k: (act_pre[act_pre.index(k)] if k in act_pre else []) for k in range(1, problem_dict['act_count']+1)}
    graph = {} #fixed graph generator
    for k in range(1, problem_dict['act_count']+1):
        if act_pre[k]['value'] == [0]:
            graph[k] = []
        else:
            graph[k] = act_pre[k]['value']
    times_dict = {d['index']: d['value'] for d in problem_dict['act_proc']}
    r_cap_dict = {d['index']: d['value'] for d in problem_dict['r_cap']}
    r_cons_dict = {tuple(d['index']): d['value'] for d in problem_dict['r_cons']}
    r_count = problem_dict['r_count']
    return graph, times_dict, r_cap_dict, r_cons_dict, r_count, problem_dict['act_pre']
    #return MRCPSP(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)


def random_key_decoder(x, reference_list):
    sorted_x = list(sorted(x))
    return [reference_list[sorted_x.index(e)] for e in x]


#https://codereview.stackexchange.com/questions/239008/python-implementation-of-kahns-algorithm
def khan(graph):
    in_degree = {u : 0 for u in graph}
    for vertices, neighbors in graph.items():
        in_degree.setdefault(vertices, 0)
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    no_indegree_vertices = {vertex for vertex, count in in_degree.items() if count == 0}

    topological_sort = []
    while no_indegree_vertices:
        vertex = no_indegree_vertices.pop()
        topological_sort.append(vertex)
        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                no_indegree_vertices.add(neighbor)

    if len(topological_sort) != len(in_degree):
        print("Graph has cycles; It is not a directed acyclic graph ... ")
        return None
    else:
        return topological_sort
        