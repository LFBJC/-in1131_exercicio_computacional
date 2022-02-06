# from DE import ourDE

print('Digite as dependências das atividades.')
print('Para fazê-lo digite pares de atividades separados por espaço com o seguinte formato:')
print('        nome_da_atividade: dependências')
print('Exemplo: atividade2:atividade1,atividade6 atividade4:atividade3 atividade5:atividade3,atividade4')
activities_string = input('Digite aqui: ')
activities_string_components = activities_string.split()
graph = [[component.split(':')[0], component.split(':')[1].split(',')] for component in activities_string_components]
dependent_tasks = [restriction[0] for restriction in graph]
graph += [[e, []] for component in activities_string_components for e in component.split(':')[1].split(',')
          if e not in dependent_tasks]
# topologically_sorted_activities = khan(graph)
