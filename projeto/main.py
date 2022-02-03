from knap import KnapSack
from DE import ourDE

items = [(3, 3), (2, 2), (1, 1), (2.2, 2), (1.4, 1), (3.8, 4), (0.2, 1), (0.1, 1), (0.13, 1),
        (2.8, 3), (1.5, 2), (2, 2), (3.1, 3), (1.2, 1), (1.7, 3), (1.1, 2), (0.3, 1)]
bags_list = [13, 9, 7]

problem = KnapSack(knapsacks=bags_list, items=items)
init_pop = len(bags_list) * len(items) * 20

ourDE.solver(problem=problem, init_pop=init_pop)
