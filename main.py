#from pymoo.algorithms.soo.nonconvex.ga import GA

from numpy.core.fromnumeric import var
from pymoo.problems.single import ackley
from pymoo.problems.single.knapsack import Knapsack
from objective_functions import Ackley, Griewank, Colville, Trid, KnapSack
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
from pymoo.operators.sampling.lhs import LHS
from pymoo.factory import get_problem
#from pymoo.problems.single.knapsack import Knapsack, MultiObjectiveKnapsack, create_random_knapsack_problem
from pymoo.factory import get_crossover, get_mutation, get_sampling

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

def q2(problem=None):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.factory import get_problem, get_sampling, get_crossover, get_mutation
    from pymoo.optimize import minimize
    from pymoo.visualization.scatter import Scatter
   
    algorithm = NSGA2(pop_size=100,                  
                    sampling=get_sampling("bin_random"),
                    crossover=get_crossover("bin_two_point"),
                    mutation=get_mutation("bin_bitflip"),
                    eliminate_duplicates=True)

    res = minimize(problem,
                algorithm,
                ('n_gen', 50),
                seed=1,
                verbose=False)
    plot = Scatter()
    plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    plot.add(res.F, facecolor="none", edgecolor="red")
    plot.show()
    # algorithm = AE(
    #     pop_size=10000,
    #     sampling=get_sampling("bin_random"),
    #     crossover=get_crossover("bin_hux"),
    #     mutation=get_mutation("bin_bitflip"),
    #     eliminate_duplicates=True)

    # res = minimize(problem,
    #             algorithm,
    #             ('n_gen', 5),
    #             verbose=True)

    print("Best solution found: %s" % res.X.astype(int))
    print("Function value: %s" % res.F)
    print("Constraint violation: %s" % res.CV)


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
                #tupla objetos(profit,weights)
                items = [(3,3),(2,2),(1,1),(2,2.2),(1,1.4),(4,3.8),(1,0.2),(1,0.1),
                (1,0.13),(3,2.8),(2,1.5),(2,2),(3,3.1),(1,1.2),(3,1.7),(2,1.1),(1,0.3)]
                knapsacks = [13,9,7]
                q2(KnapSack(knapsacks=knapsacks, items=items))
            elif padrao == '2':
                print("Opção indisponível")
                # bags = input("Digite o numero de mochilas:\n")
                # print(f'You entered {bags}')
                # objs = input("Digite o numero de objetos:\n")
                # print(f'You entered {objs}')
                #q2(KnapSack(bags, objs))
        else:
            print("Opção não disponível")
            exit()