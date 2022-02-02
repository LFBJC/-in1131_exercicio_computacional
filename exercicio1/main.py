from pymoo.algorithms.soo.nonconvex.ga import GA

# from np.core.fromnumeric import var
# from pymoo.problems.single import ackley
# from pymoo.problems.single.knapsack import Knapsack
import time

from eda_definition import run_eda_instance
from objective_functions import Ackley, Griewank, Colville, Trid, KnapSack
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import numpy as np
from utils import t_test
from tqdm import tqdm
# from pymoo.operators.sampling.lhs import LHS

# from pymoo.factory import get_problem
# from pymoo.problems.single.knapsack import Knapsack, MultiObjectiveKnapsack, create_random_knapsack_problem
# from pymoo.factory import get_crossover, get_mutation, get_sampling
import matplotlib.pyplot as plt
import numpy as np

from pymoo.core.callback import Callback


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(algorithm.pop.get("F").min())

class KnapCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        self.data["best"].append(-1 * algorithm.pop.get("F").min())

def q1(problem=Ackley(), include_eda=False):
    iterations = 10# 1000  #10000

    print(problem.name)
    es = ES(callback=MyCallback())
    de = DE(callback=MyCallback())
    ga = GA(callback=MyCallback())
    es_results = []
    de_results = []
    ga_results = []
    eda_results = []

    #criterion = ("n_eval", 10000)
    criterion = ("n_gen", 100)
    for _ in tqdm(range(iterations)):
        aux_es =  minimize(problem, es, criterion)
        es_results.append(aux_es.F)
        aux_de =  minimize(problem, de, criterion)
        de_results.append(aux_de.F)
        aux_de =  minimize(problem, de, criterion)
        aux_ga = minimize(problem, ga, criterion)
        ga_results.append(aux_ga.F)
        if include_eda:
            aux_eda, res_eda, fit_mins = run_eda_instance(problem, ngen=criterion[1])
            eda_results.append(aux_eda)
    
    print('resultados para Estrategia Evolutiva:')
    print('      solucao', aux_es.X)
    print('      media:', np.mean(es_results))
    print('      desvio:', np.std(es_results))
    val = aux_es.algorithm.callback.data["best"]
    plt.plot(np.arange(len(val)), val, marker='', markerfacecolor='blue', label='ES')

    print('resultados para Evolucao Diferencial:')
    print('      solucao', aux_de.X)
    print('      media:', np.mean(de_results))
    print('      desvio:', np.std(de_results))
    val = aux_de.algorithm.callback.data["best"]
    plt.plot(np.arange(len(val)), val, marker='', markerfacecolor='orange', label='DE')

    print('resultados para Algoritmos Genéticos:')
    print('      solucao', aux_ga.X)
    print('      media:', np.mean(ga_results))
    print('      desvio:', np.std(ga_results))

    val = aux_ga.algorithm.callback.data["best"]
    plt.plot(np.arange(len(val)), val, marker='', markerfacecolor='green', label='GA')

    if include_eda:
        print('resultados para Algoritmos de Estimação de Distribuição:')
        print('      solucao', res_eda[0])    
        print('      media:', np.mean(eda_results))
        print('      desvio:', np.std(eda_results))
        
        plt.plot(np.arange(len(fit_mins)), fit_mins, marker='', markerfacecolor='red', label='EDA')
    plt.legend()
    plt.title("GA x ES x ED x EDA for " + problem.name )
    plt.savefig("comp_" + problem.name + ".png")
    plt.clf()
    best, best_results = t_test(['ES', 'DE'], results=[es_results, de_results])
    best, best_results = t_test(['GA', best], results=[ga_results, best_results])
    if include_eda:
        best, best_results = t_test(['EDA', best], results=[eda_results, best_results])
    print('O melhor algoritmo encontrado para o problema solicotado foi: '+best)
    print('\n')

    from pymoo.factory import get_problem, get_visualization

    if problem.n_var == 2:
        get_visualization("fitness-landscape", problem, angle=(45, 45), _type="surface").show()
        get_visualization("fitness-landscape", problem, _type="contour", colorbar=True).show()


def q2(knapsacks=None, items=None):
    from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation, get_selection
    from pymoo.optimize import minimize
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.operators.mutation.bitflip import BinaryBitflipMutation

    problem = KnapSack(knapsacks=knapsacks, items=items)
    init_pop = len(knapsacks) * len(items) * 20
    algorithm = GA(
        pop_size=init_pop, #pensar numa estrategia de ajustar de acordo com o numero de items * mochila!
        sampling=get_sampling("bin_random"),
        crossover=get_crossover("bin_exp"), #bin_hux
        mutation=get_mutation("bin_bitflip"),
        selection = get_selection('random'),
        eliminate_duplicates=True)
    start_time = time.time()
    res = minimize(problem,
                   algorithm,
                   #('n_eval', 10000*5),
                   ('n_gen', 100),
                   verbose=True,
                   callback=KnapCallback())
    print("Items (volume,valor): %s\n" % str(items))

    if res.X is not None:
        num = 0
        chunks = [res.X[x:x+len(items)] for x in range(0, len(res.X), len(items))]
        result = [0,0]
        for i in knapsacks:
            num +=1
            print("Mochila " + str(num) + "(" + str(i) + " u.v): ")
            indice = 0
            moch_aux = []
            volume_total = 0
            val_total = 0
            for obj in chunks[num-1]:
                if obj == True:
                    moch_aux.append(items[indice])
                    volume_total += items[indice][0]
                    val_total += items[indice][1]
                indice += 1
            print(str(moch_aux) + " - volume na mochila: " + str(round(volume_total,2)) + 
                    " / valor na mochila: " + str(round(val_total,2)) + "\n")
            result[0] += volume_total
            result[1] += val_total
        print("Volume somado: %s\nValor somado: " % result[0], result[1])

        print("\nBest solution found: ")
        print(problem.x_to_matrix(res.X))

        print("Function value: %s" % res.F)
        print("Constraint violation: %s" % res.CV)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("\n\n\n")

        val = res.algorithm.callback.data["best"]
        plt.plot(np.arange(len(val)), val, marker='', markerfacecolor='green', label='knapsack')

        plt.legend()
        plt.title("Fitness x Generation")
        plt.savefig("comp_knapsack.png")

    else:
        print("Error: Could not find a solution")


if __name__ == "__main__":

    while True:
        value = input("Digite uma opção (ackley, griewank, colville, trid ou knapsack):\n")
        print(f'You entered {value}\n')
        if value == 'ackley':
            eda = input(
                'Deseja incluir Algoritmos de Estimação de Distribuição (EDA) no teste? (1 para sim 0 para não)\n')
            if eda == '1':
                q1(Ackley(), include_eda=True)
            else:
                q1(Ackley(), include_eda=False)
        elif value == 'griewank':
            eda = input(
                'Deseja incluir Algoritmos de Estimação de Distribuição (EDA) no teste? (1 para sim 0 para não)\n')
            if eda == '1':
                q1(Griewank(), include_eda=True)
            else:
                q1(Griewank(), include_eda=False)
        elif value == 'colville':
            eda = input(
                'Deseja incluir Algoritmos de Estimação de Distribuição (EDA) no teste? (1 para sim 0 para não)\n')
            if eda == '1':
                q1(Colville(), include_eda=True)
            else:
                q1(Colville(), include_eda=False)
        elif value == 'trid':
            eda = input(
                'Deseja incluir Algoritmos de Estimação de Distribuição (EDA) no teste? (1 para sim 0 para não)\n')
            if eda == '1':
                q1(Trid(), include_eda=True)
            else:
                q1(Trid(), include_eda=False)
        elif value == 'knapsack':
            padrao = input("Digite 1 p/ resolver a Q2 (17 objetos e 3 mochilas), digite 2 para um problema diferente: \n")
            if padrao == '1':
                #tupla objetos(weights,profit)
                items = [(3, 3), (2, 2), (1, 1), (2.2, 2), (1.4, 1), (3.8, 4), (0.2, 1), (0.1, 1), (0.13, 1),
                         (2.8, 3), (1.5, 2), (2, 2), (3.1, 3), (1.2, 1), (1.7, 3), (1.1, 2), (0.3, 1)]
                bags_list = [13, 9, 7]

                q2(knapsacks=bags_list, items=items)
            elif padrao == '2':
                items = []
                val_input = input("Digite os valores dos items separado por espaço:\n")
                val_list = val_input.split()
                print(f'You entered {val_list}\n')
                volume_input = input("Digite os volumes dos items separado por espaço:\n")
                volume_list = volume_input.split()
                print(f'You entered {volume_list}\n')
                for i in range(len(val_list)):
                    items.append((int(volume_list[i]),int(val_list[i])))
                print(f'You entered {items}\n')
                bags = input("Digite a capacidade das mochilas separado por espaço:\n")
                bags_list = bags.split()
                for j in range(len(bags_list)):
                    bags_list[j] = int(bags_list[j])
                q2(knapsacks=bags_list, items=items)
        else:
            print("Opção não disponível")
            exit()