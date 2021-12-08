from scipy.stats import ttest_ind
import numpy as np


def t_test(names, results):
    stat, pvalue = ttest_ind(results[0], results[1])
    print('Teste T para os algoritmos '+names[0]+' e '+names[1])
    print('    P-valor encontrado:', pvalue[0])
    if pvalue < 0.05:
        if np.mean(results[0]) > np.mean(results[1]):
            print('    O algoritmo ' + names[0] + ' obteve resultados melhores com nível de confiança 95%')
        else:
            print('    O algoritmo ' + names[1] + ' obteve resultados melhores com nível de confiança 95%')
    else:
        print('    Para um nível de significância de 5% os resultados são equivalentes')
