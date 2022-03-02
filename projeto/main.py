# from DE import ourDE
# from utils import khan
from objective import RCPSP, RCPSP_RandomKeyRepresentation, RCPSP_RKR_debug
from utils import problem_from_json
import sys, os, json, math

import pandas as pd
from tqdm import tqdm
import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from operators import SamplingRespectingPrecedence, SamplingWithSelection
import glob
import matplotlib.pyplot as plt
from mutation import *

CRITERION = ("n_gen", 1000)
ITERATIONS = 30

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
# just for debugging
# SEED = 268567135
# np.random.seed(SEED)
#----------------versão nova---------------#
while True:
    print("\nenter 'ctrl + c' to finish")
    #jobs = input("\nSelecione o número de jobs (30, 60, 90 ou 120):\n" )
    jobs = 30
    if int(jobs) == 120:
        aux = input("Selecione um número entre 1 e 60:\n" )
    else:
        aux = input("Selecione um número entre 1 e 48:\n" )
    aux2 = input("Selecione um número entre 1 e 10:\n" )
    #aux2=8
    file_name = str(os.getcwd()) + "/data/instances/json/j" + str(jobs) + '/j' + str(jobs) + str(aux) +  '_' +str(aux2) + '.json'

    instance = 'j' + str(jobs) + str(aux) +  '_' + str(aux2)
    sampling_tipe = 'standard' #SamplingWithSelection, SamplingRespectingPrecedence or standard
    representation = 'RK'
    mutation = 'agressive'
    if os.path.isfile(file_name):
        for _ in tqdm(range(ITERATIONS)):
            graph, times_dict, r_cap_dict, r_cons_dict, r_count, act_pre = problem_from_json(file_name)
            if representation == 'RK':
                problem = RCPSP_RandomKeyRepresentation(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict, r_count=r_count, act_pre=act_pre)
            elif representation == 'start_times':
                problem = RCPSP(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict)
            # problem = RCPSP_RKR_debug(graph=graph, times_dict=times_dict, r_cap_dict=r_cap_dict, r_cons_dict=r_cons_dict, r_count=r_count, act_pre=act_pre)

            pop= math.exp(3.551 + (22.72/jobs))
            if sampling_tipe == 'standard':
                de = DE(pop_size=int(pop), mutation=get_mutation("bitflip", prob=0.30)) 
            if sampling_tipe == 'SamplingWithSelection':
                de = DE(pop_size=int(pop), sampling=SamplingWithSelection())  # DE(sampling=SamplingRespectingPrecedence(pop_ratio=0.8, max_depth=30))            
            if sampling_tipe == 'SamplingRespectingPrecedence':
                de = DE(pop_size=int(pop), sampling=SamplingRespectingPrecedence(pop_ratio=1, max_depth=30))  # DE(sampling=SamplingRespectingPrecedence(pop_ratio=0.8, max_depth=30))            
            
            res = minimize(problem, de, CRITERION, save_history=True,               
                        verbose=True)

            if res.F is not None:
                x_results.append(res.X)
                fitness_results.append(res.F)
            else:
                none_count += 1

        fitness_results = [f for f in fitness_results if f is not None]

        #df_res = pd.read_csv(str(os.getcwd()) + '/data/solutions/' + 'j' + str(jobs) + '.csv', sep=';')

        path = str(os.getcwd()) + '/data/solutions/' + 'j' + str(jobs) + '.csv' # use your path
        
        df_res = pd.read_csv(path, index_col=None, header=0, sep=',')
        print(df_res.head())

        new_row = {'instance': instance,  'min_makespan':min(fitness_results), 'average_makespan':np.mean(fitness_results), 
                     'std_makespan':np.mean(fitness_results), 'sampling_tipe':sampling_tipe, 'representation': representation, 'mutation': mutation}
        #append row to the dataframe
        df_res.loc[-1] = ['', instance, fitness_results ,min(fitness_results)[0], np.mean(fitness_results), np.std(fitness_results), sampling_tipe, representation, mutation]  # adding a row
        df_res.index = df_res.index + 1  # shifting index
        df_res = df_res.sort_index()  # sorting by index

        df_res = df_res.reindex(columns=['instance', 'min_makespan', 'average_makespan', 'std_makespan', 'sampling_tipe', 'representation'])
        df_res.to_csv(str(os.getcwd()) + '/data/solutions/' + 'j' + str(jobs) + '.csv')

        filename_save = ''.join("{}_{}".format(k, v) for k, v in new_row.items())
        #print(str(res.history), file=open(str(os.getcwd()) + '/data/solutions/' + filename_save +'.txt', "a"))
        
        fig = plt.figure(figsize=(3, 6))
        val = [e.opt.get("F")[0] for e in res.history]
        plt.plot(np.arange(len(val)), val)
        #plt.show()
        fig.savefig(str(os.getcwd()) + '/data/solutions/j' + str(jobs) + '/' + filename_save + '.png', dpi=fig.dpi)

        #np.savetxt(str(os.getcwd()) + '/data/solutions/' + 'j' + str(jobs) + '/' + str(new_row) +'.txt', res.history , delimiter=",")

        print('resultados:')
        print('      Número de vezes que não foi encontrada nenhuma solução viável:', none_count)
        print('      Média (para os casos em que soluções foram encontradas):', np.mean(fitness_results))
        print('      Desvio (para os casos em que soluções foram encontradas):', np.std(fitness_results))
    else:
        print("\nProblema não existe, digite uma opção válida.")
    break
