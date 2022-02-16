# from DE import ourDE
# from utils import khan
from objective import RCPSP, RCPSP_RandomKeyRepresentation
from utils import problem_from_json
import sys, os, json

from tqdm import tqdm
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

CRITERION = ("n_gen", 500)
ITERATIONS = 1

#----------------versão antiga onde é preciso digitar os inputs---------------#
# print('Digite as dependências das atividades.')
# print('Para fazê-lo digite pares de atividades separados por espaço com o seguinte formato:')
# print('        nome_da_atividade: dependências')
# print('Exemplo: atividade2:atividade1,atividade6 atividade4:atividade3 atividade5:atividade3,atividade4')
# # activities_string = input('Digite aqui: ')
# activities_string_components = activities_string.split()
# graph = {component.split(':')[0]: component.split(':')[1].split(',') for component in activities_string_components}
# all_nodes = list(set([task for k, v in graph.items() for task in [k]+v]))
# times_dict = {}
# for node in all_nodes:
#     times_dict[node] = float(input("Digite o tempo necessário para realizar a atividade "+node+" (um número real):"))
# dependent_tasks = list(graph.keys())
# not_dependent = [node for node in all_nodes if node not in dependent_tasks]
# for task in not_dependent:
#     graph[task] = []
x_results = []
fitness_results = []
none_count = 0
#----------------versão nova---------------#
while True:
    print("\nenter 'ctrl + c' to finish")
    jobs = input("\nSelecione o número de jobs (30, 60, 90 ou 120):\n" )
    if int(jobs) == 120:
        aux = input("Selecione um número entre 1 e 60:\n" )
    else:
        aux = input("Selecione um número entre 1 e 48:\n" )
       
    aux2 = input("Selecione um número entre 1 e 10:\n" )
    file_name = str(os.getcwd()) + "/data/instances/json/j" + str(jobs) + '/j' + str(jobs) + str(aux) +  '_' +str(aux2) + '.json'
    if os.path.isfile(file_name):
        for _ in tqdm(range(ITERATIONS)):
            graph, times_dict, r_cap_dict, r_cons_dict  = problem_from_json(file_name)
            #problem = RCPSP(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)
            problem = RCPSP_RandomKeyRepresentation(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)
            de = DE()
            res = minimize(problem, de, CRITERION)
            if res.F is not None:
                x_results.append(res.X)
                fitness_results.append(res.F)
            else:
                none_count += 1

        fitness_results = [f for f in fitness_results if f is not None]
        print('resultados:')
        print('      Número de vezes que não foi encontrada nenhuma solução viável:', none_count)
        print('      Média (para os casos em que soluções foram encontradas):', np.mean(fitness_results))
        print('      Desvio (para os casos em que soluções foram encontradas):', np.std(fitness_results))
    else:
        print("\nProblema não existe, digite uma opção válida.")
