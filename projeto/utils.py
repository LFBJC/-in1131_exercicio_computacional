import json
import numpy as np
import bisect
from copy import copy
import time


def problem_from_json(file_name):
    with open(file_name) as json_file:
        problem_dict = json.load(json_file)
    act_pre = problem_dict['act_pre']
    #graph = {k: (act_pre[act_pre.index(k)] if k in act_pre else []) for k in range(1, problem_dict['act_count']+1)}
    graph = {} #fixed graph generator
    for k in range(0, problem_dict['act_count']+1):
        graph[k] = act_pre[k]['value']
    times_dict = {d['index']: d['value'] for d in problem_dict['act_proc']}
    r_cap_dict = {d['index']: d['value'] for d in problem_dict['r_cap']}
    r_cons_dict = {tuple(d['index']): d['value'] for d in problem_dict['r_cons']}
    r_count = problem_dict['r_count']
    times_dict[0] = 0
    return graph, times_dict, r_cap_dict, r_cons_dict, r_count, problem_dict['act_pre']
    #return MRCPSP(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)


def khan(graph):
    ret = []
    no_incoming_edge = [key for key in graph.keys() if graph[key] == []]
    while no_incoming_edge != []:
        current_node, no_incoming_edge = no_incoming_edge[0], no_incoming_edge[1:]
        ret.append(current_node)
        for other_node in graph.keys():
            if other_node != current_node and current_node in graph[other_node]:
                graph[other_node] = [node for node in graph[other_node] if node != current_node]
                if graph[other_node] == []:
                    no_incoming_edge.append(other_node)
    edges = [(k, x) for k in graph.keys() if graph[k] != [] for x in graph[k]]
    if edges != []:
        raise ValueError("The graph has cycles")
    else:
        return ret


def solution_from_random_key(indiv, graph):
    # ret = []
    prioritize_it = len(indiv)+max([len(v) for v in graph.values()])
    sorting_function = lambda t: prioritize_it*(graph[t] == []) + indiv[-1]*len(graph[t]) + indiv[list(graph.keys()).index(t)]
    sorted_keys = [k for k in sorted(graph.keys(), key=sorting_function)]
    """
    # I tried to modify in order to turn it more random (which required an increase on the length of the individuals)
    # but it didn't work 
    no_incoming_edge = sorted([k for k in graph.keys() if graph[k] == []], key=lambda t: indiv[list(graph.keys()).index(t)])
    indiv_index = len(no_incoming_edge)
    while no_incoming_edge != []:
        current_node, no_incoming_edge = no_incoming_edge[0], no_incoming_edge[1:]
        ret.append(current_node)
        nodes_with_an_incoming_edge = sorted([n for n in graph.keys() if current_node in graph[n]], key=lambda t: indiv[indiv_index+list(graph.keys()).index(t)])
        indiv_index += len(nodes_with_an_incoming_edge)
        for other_node in nodes_with_an_incoming_edge:
            graph[other_node] = [node for node in graph[other_node] if node != current_node]
            if graph[other_node] == []:
                no_incoming_edge.append(other_node)
    edges = [(k, x) for k in graph.keys() if graph[k] != [] for x in graph[k]]
    if edges != []:
        raise ValueError("The graph has cycles")
    else:
        return ret
    """
    return sorted_keys


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

def get_start_time(preact, solution):
    preact_start_time = None
    for dic in solution:    #Pegar o tempo de inicio
        for act, start_time in dic.items():
            if act == preact:
                return start_time

def activity_in_conflict_in_precedence(act_pre, activity, time_unit, times_dict, solution):
    #essa funcao ta errada? - fix
    actives_in_sol = [list(dic.keys())[0] for dic in solution]

    if act_pre[activity]['value'] == []:
        return False

    for preact in act_pre[activity]['value']:
        if preact in actives_in_sol: #Se a preatividade existe,continue
            if get_start_time(preact, solution) + times_dict[preact] > time_unit:
                return True
    else:
        return False

# print(activity)
# print(act_pre[activity]['value'])
# import time; time.sleep(60)


def is_resource_usage_greater_than_supply(r_count, actual_resource_usage):
    for resource in range(1, r_count+1):
        if actual_resource_usage[resource] is None:
            continue
        if list(actual_resource_usage[resource])[0] < list(actual_resource_usage[resource])[1]:
            #print(list(actual_resource_usage[resource]))
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

def another_update_resource_usages_in_time(resource_usages, activity, r_cons_dict, times_dict, list_start_time, ind, graph):    
    for i, activity in enumerate(ind):
        if graph[activity] != []:
            list_start_time[i] = max([list_start_time[other] + times_dict[other] for other in graph[activity]])
        for resource in resource_usages.keys():
            if (activity, resource) in r_cons_dict.keys():
                resource_usages[resource][list_start_time[i]:list_start_time[i]+times_dict[activity]] += r_cons_dict[(activity, resource)]
                
    for resource in resource_usages.keys():
        if (activity, resource) in r_cons_dict.keys():
            resource_usages[resource][list_start_time[i]:list_start_time[i]+times_dict[activity]] += r_cons_dict[(activity, resource)]
    return resource_usages
    
def insert_value_to_ordered_list(l, value):
    i = bisect.bisect_left(l, value)
    if i >= len(l) or not l[i] == value:
        l.insert(i,value)
    return l


def serialSGS(ind, total_time_all_activit, r_count, r_cons_dict, r_cap_dict, times_dict, act_pre, graph, resource_usages):
    #INICIO Serial SGS para todos os individuos
    solution = []
    resource_usages_in_time = {}
    time_points = [0]
    for sec in range(total_time_all_activit):
        resource_usages_in_time[sec] = {}
        for key, value in r_cap_dict.items():
            key_aux = {key: (value,0)} #tupla capacidade/em uso
            resource_usages_in_time[sec].update(key_aux)
    
    solution.append({0: 0})  #inicializando dummy
    ind = [activity for activity in ind if activity > 0]
    for activity in ind:
        list_start_time = [0]*(len(ind)+1)
        activity = int(activity)
        last_time = time_points[-1]
        start_time = 0
        for time_unit in reversed(time_points):
            actual_resource_usage = copy(resource_usages_in_time[time_unit])
            actual_resource_usage = add_resource_usage(actual_resource_usage, r_count, r_cons_dict, activity)   
            is_resourc_enough = is_resource_usage_greater_than_supply(r_count, actual_resource_usage)
            is_conflit_precedence = activity_in_conflict_in_precedence(act_pre, activity, time_unit, times_dict, solution)
            if is_resourc_enough or is_conflit_precedence:
                    start_time = last_time
                    break
            else:
                last_time = time_unit
        tuple_start_time = {activity: start_time}
        list_start_time[activity] = start_time
        # print('\n')
        # print(tuple_start_time)
        # print(resource_usages_in_time[start_time])
        # print('\n')
        solution.append(tuple_start_time)
        time_points = insert_value_to_ordered_list(time_points, start_time)
        time_points = insert_value_to_ordered_list(time_points, start_time + times_dict[activity])   
        
        resource_usages_in_time = update_resource_usages_in_time(resource_usages_in_time, activity, start_time, times_dict, r_count, r_cons_dict)
        resource_usages_ret = another_update_resource_usages_in_time(resource_usages, activity, r_cons_dict, times_dict, list_start_time, ind, graph)
    # print('\n==============================\n')
    # print(resource_usages_in_time)
    # print('\n==============================\n')
    # print(activity)
    # print(start_time)
    # print(solution)
    # time.sleep(6000)
    return solution, resource_usages_ret


def compute_makespan(solution, times_dict):
    sol_sorted_by_values = sorted(solution, key=lambda d: list(d.values())) 
    return list(sol_sorted_by_values[-1].items())[0][1] + times_dict[list(sol_sorted_by_values[-1].items())[0][0]]


def check_if_solution_feasible(solution, times_dict, r_cap_dict, r_count, r_cons_dict, act_pre):
    makespan = compute_makespan(solution, times_dict)
    #total_time_all_activit = sum(value for key, value in (times_dict.items()))
    resource_usage = {}
    for sec in range(makespan+1):
        resource_usage[sec] = {}
        for key, value in r_cap_dict.items():
            key_aux = {key: (value,0)} #tupla capacidade/em uso
            resource_usage[sec].update(key_aux)
    for i in range(makespan):
        for dic in solution:
            for activity, start_time in dic.items():
                if start_time <= i < start_time + times_dict[activity]:
                    resource_usage[i] = add_resource_usage(resource_usage[i], r_count, r_cons_dict, activity) 
        if is_resource_usage_greater_than_supply(r_count, resource_usage[i]):
            return 1.0
    for dic in solution:
        for activity, start_time in dic.items():
            for sec in range(makespan+1):
                if activity_in_conflict_in_precedence(act_pre, activity, sec, times_dict, solution):
                    return 1.0
                else:    
                    return -1.0


    