import json
import numpy as np
import bisect

def problem_from_json(file_name):
    with open(file_name) as json_file:
        problem_dict = json.load(json_file)
    act_pre = problem_dict['act_pre']
    #graph = {k: (act_pre[act_pre.index(k)] if k in act_pre else []) for k in range(1, problem_dict['act_count']+1)}
    graph = {} #fixed graph generator
    for k in range(0, problem_dict['act_count']+1):
        if k == 0:
            continue
        graph[k] = act_pre[k]['value']
    times_dict = {d['index']: d['value'] for d in problem_dict['act_proc']}
    r_cap_dict = {d['index']: d['value'] for d in problem_dict['r_cap']}
    r_cons_dict = {tuple(d['index']): d['value'] for d in problem_dict['r_cons']}
    r_count = problem_dict['r_count']
    times_dict[0] = 0
    return graph, times_dict, r_cap_dict, r_cons_dict, r_count, problem_dict['act_pre']
    #return MRCPSP(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)


#https://codereview.stackexchange.com/questions/239008/python-implementation-of-kahns-algorithm
def khan(graph):
    in_degree = {u : 0 for u in graph}
    out_degree = {u : 0 for u in graph}
    for vertices, neighbors in graph.items():
        in_degree.setdefault(vertices, 0)
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
            out_degree[vertices] = out_degree.get(neighbor, 0) + 1
    no_indegree_vertices = {vertex for vertex, count in in_degree.items() if count == 0}
    no_outdegree_vertices = {vertex for vertex, count in out_degree.items() if count == 0}

    topological_sort = []
    while no_indegree_vertices:
        vertex = no_indegree_vertices.pop()
        topological_sort.append(vertex)
        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                no_indegree_vertices.add(neighbor)

    assert all([topological_sort.index(v) < topological_sort.index(k) for k in graph.keys() for v in graph[k]])
    if len(topological_sort) != len(in_degree):
        print("Graph has cycles; It is not a directed acyclic graph ... ")
        return None
    else:
        return topological_sort, no_outdegree_vertices


def random_key_decoder(x, reference_list):
    sorted_x = list(sorted(x))
    return [reference_list[sorted_x.index(e)] for e in x]


def another_random_key_decoder(indvs, reference_list):
    for arr in range(len(indvs)):
        ordenate_x = np.sort(indvs[arr])
        for i in range(len(ordenate_x)):
            idx, = np.where(indvs[arr] == ordenate_x[i])
            indvs[arr][idx[0]] = i
    return indvs

def activity_in_conflict_in_precedence(act_pre, activity, time_unit, times_dict, solution):
    if act_pre[activity]['value'] == []:
        return False
    for preact in act_pre[activity]['value']:
        if not solution:
            return True
        else:
            for dic in solution:
                if preact in dic.keys():
                    if (dic[preact] + times_dict[preact] > time_unit):
                        return True
                    else:
                        return False
                else:
                    return True

def is_resource_usage_greater_than_supply(r_count, actual_resource_usage):
    for resource in range(1, r_count+1):
        if actual_resource_usage[resource] is None:
            continue
        if list(actual_resource_usage[resource])[0] < list(actual_resource_usage[resource])[1]:
            return True
        else:
            return False

def add_resource_usage(actual_resource_usage, r_count, r_cons_dict, activity):
    #add resource usage
    for resource in range(1, r_count+1):
        #add_resource_usage
        y = list(actual_resource_usage[resource])
        if activity == 0:
            y[1] = y[1] 
        else:
            y[1] = y[1] + r_cons_dict[(activity, resource)]
        actual_resource_usage[resource] = tuple(y) 
    return actual_resource_usage        


def update_resource_usages_in_time(resource_usages_in_time, activity, start_time, times_dict, r_count, r_cons_dict):
    for point in range(start_time + times_dict[activity]):
        #actual_resource_usage = resource_usages_in_time[point]
        resource_usages_in_time[point] = add_resource_usage(resource_usages_in_time[point], r_count, r_cons_dict, activity)
    return resource_usages_in_time



def insert_value_to_ordered_list(l, value):
    i = bisect.bisect_left(l, value)
    if i >= len(l) or not l[i] == value:
        l.insert(i,value)
    return l

def compute_makespan():
    return 1


def check_if_solution_feasible(solution):
    makespan = compute_makespan(solution)
    for i in range(makespan):
        resource_usage = ResourceUsage()
        for activity, start_time in solution.iteritems():
            if start_time <= i < start_time + activity.duration:
                resource_usage.add_resource_usage(activity.demand)
        
        if resource_usage.is_resource_usage_greater_than_supply( self.resources):
            return False
    return True
    