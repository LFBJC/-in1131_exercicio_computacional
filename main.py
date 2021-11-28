from pymoo.problems.single import ackley
from objective_functions import Ackley, Griewank, Colville, Trid, KnapSack
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm
from pymoo.operators.sampling.lhs import LHS
from pymoo.factory import get_problem


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

def q2():
    pass

if __name__ == "__main__":
    
    value = input("Digite uma opção (ackley, griewank, colville, trid ou knapsack):\n")
    print(f'You entered {value}')
    if value == 'ackley':
        q1(Ackley())
    elif value == 'griewank':
        q1(Griewank())
    elif value == 'colville':
        q1(Colville())
    elif value == 'trid':
        q1(Trid())
    elif value == 'knapsack':
        bags = input("Digite o numero de mochilas:\n")
        print(f'You entered {bags}')
        objs = input("Digite o numero de objetos:\n")
        print(f'You entered {objs}')
        #q2(KnapSack(bags, objs))
    else:
        print("Opção não disponível")
        exit()