from objective_functions import Ackley, Griewank
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm


iterations = 30  # 1000  #10000
problem_names = ['Ackley', 'Griewank']
problems = [Ackley(), Griewank()]
for problem, name in zip(problems, problem_names):
    print(name)
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
