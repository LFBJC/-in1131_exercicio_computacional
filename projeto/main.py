# from DE import ourDE
from utils import khan

print('Digite as dependências das atividades.')
print('Para fazê-lo digite pares de atividades separados por espaço com o seguinte formato:')
print('        nome_da_atividade: dependências')
print('Exemplo: atividade2:atividade1,atividade6 atividade4:atividade3 atividade5:atividade3,atividade4')
activities_string = input('Digite aqui: ')
activities_string_components = activities_string.split()
graph = [[component.split(':')[0], component.split(':')[1].split(',')] for component in activities_string_components]
dependent_tasks = [restriction[0] for restriction in graph]
all_nodes = list(set([task for c in graph for task in [c[0]]+c[1]]))
not_dependent = [node for node in all_nodes if node not in dependent_tasks]
graph += [[task, []] for task in not_dependent]
topologically_sorted_activities = khan(graph)
print(topologically_sorted_activities)
