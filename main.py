from pymoo.algorithms.soo.nonconvex.ga import GA

# from numpy.core.fromnumeric import var
# from pymoo.problems.single import ackley
# from pymoo.problems.single.knapsack import Knapsack
from objective_functions import Ackley, Griewank, Colville, Trid, KnapSack
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
# from pymoo.operators.sampling.lhs import LHS

# from pymoo.factory import get_problem
# from pymoo.problems.single.knapsack import Knapsack, MultiObjectiveKnapsack, create_random_knapsack_problem
# from pymoo.factory import get_crossover, get_mutation, get_sampling

def q1(problem=Ackley()):
    iterations = 30  # 1000  #10000

    print(problem.name)
    es = ES()
    de = DE()
    es_results = []
    de_results = []
    for i in tqdm(range(iterations)):
        es_results.append(minimize(problem, es, ("n_eval", 10000)).F)
        de_results.append(minimize(problem, de, ("n_eval", 10000)).F)
    print('resultados para Estrategia Evolutiva:')
    print('      media:', np.mean(es_results))
    print('      desvio:', np.var(es_results))
    print('resultados para Evolucao Diferencial:')
    print('      media:', np.mean(de_results))
    print('      desvio:', np.var(de_results))
    stat, pvalue = ttest_ind(es_results, de_results)
    print('Estatistica do teste T para estes dois algoritmos:', stat)
    print('P-valor encontrado:', pvalue)
    print('\n')

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

    res = minimize(problem,
                algorithm,
                ('n_eval', 100000),
                verbose=True)
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
        print("\n\n\n")
    else:
        print("Error: Could not find a solution")

if __name__ == "__main__":

    while True:
        value = input("Digite uma opção (ackley, griewank, colville, trid ou knapsack):\n")
        print(f'You entered {value}\n')
        if value == 'ackley':
            q1(Ackley())
        elif value == 'griewank':
            q1(Griewank())
        elif value == 'colville':
            q1(Colville())
        elif value == 'trid':
            q1(Trid())
        elif value == 'knapsack':
            padrao = input("Digite 1 p/ resolver a Q2 (17 objetos e 3 mochilas), digite 2 para um problema diferente: \n")
            if padrao == '1':
                #tupla objetos(weights,profit)
                items = [(3, 3), (2, 2), (1, 1), (2.2, 2), (1.4, 1), (3.8, 4), (0.2, 1), (0.1, 1), (0.13, 1),
                        (2.8, 3), (1.5, 2), (2, 2), (3.1, 3), (1.2, 1), (1.7, 3), (1.1, 2), (0.3, 1)]
                bags_list = [13,9,7]

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