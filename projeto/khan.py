def top_sort(adj_list):
    # Find number of incoming edges for each vertex
    in_degree = {}
    for x, neighbors in adj_list.items():
        in_degree.setdefault(x, 0)
        for n in neighbors:
            in_degree[n] = in_degree.get(n, 0) + 1

    # Iterate over edges to find vertices with no incoming edges
    empty = {v for v, count in in_degree.items() if count == 0}

    result = []
    while empty:
        # Take random vertex from empty set
        v = empty.pop()
        result.append(v)

        # Remove edges originating from it, if vertex not present
        # in adjacency list use empty list as neighbors
        for neighbor in adj_list.get(v, []):
            in_degree[neighbor] -= 1

            # If neighbor has no more incoming edges add it to empty set
            if in_degree[neighbor] == 0:
                empty.add(neighbor)

    if len(result) != len(in_degree):
        return None # Not DAG
    else:
        return result

# ADJ_LIST = {
#     1: [2],
#     2: [3],
#     4: [2],
#     5: [3]
# }

ADJ_LIST = {
    3: [8, 10],
    5: [11],
    7: [8, 11],
    8: [9],
    11: [2, 9, 10],
    
    }


print(top_sort(ADJ_LIST))