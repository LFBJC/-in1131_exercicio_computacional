# from DE import ourDE
# from utils import khan
from objective import MRCPSP

from tqdm import tqdm
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

CRITERION = ("n_gen", 100)
ITERATIONS = 30
print('Digite as dependências das atividades.')
print('Para fazê-lo digite pares de atividades separados por espaço com o seguinte formato:')
print('        nome_da_atividade: dependências')
print('Exemplo: atividade2:atividade1,atividade6 atividade4:atividade3 atividade5:atividade3,atividade4')
activities_string = input('Digite aqui: ')
activities_string_components = activities_string.split()
graph = {component.split(':')[0]: component.split(':')[1].split(',') for component in activities_string_components}
all_nodes = list(set([task for k, v in graph.items() for task in [k]+v]))
times_dict = {}
for node in all_nodes:
    times_dict[node] = float(input("Digite o tempo necessário para realizar a atividade "+node+" (um número real):"))
dependent_tasks = list(graph.keys())
not_dependent = [node for node in all_nodes if node not in dependent_tasks]
for task in not_dependent:
    graph[task] = []
x_results = []
fitness_results = []
none_count = 0
for _ in tqdm(range(ITERATIONS)):
    problem = MRCPSP(graph=graph, times_dict=times_dict)
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
