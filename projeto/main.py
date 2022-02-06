# from DE import ourDE

print('Digite as dependências das atividades.')
print('Para fazê-lo digite pares de atividades separados por espaço com o seguinte formato:')
print('        nome_da_atividade: dependências')
print('Exemplo: atividade2:atividade1 atividade4:atividade3 atividade5:atividade3,atividade4')
activities_string = input()
activities_string_components = activities_string.split()
graph = [[component.split(':')[0], component.split(':')[1].split(',')] for component in activities_string_components]
# topologically_sorted_activities = khan(graph)
