.json files are processed versions of .sm files

Each item in the file means:

act_count: number of jobs (without source and sink)

r_count: number of renewable resources

act_pre: precedence relations; "index" is the index of activity and "value" is a list with the index of the precedences activities

r_cap: "index" is the index of resource, "value" is  capacity of renewable resources

r_cons: "index" is a list with [idx_activity, idx_resource] where activity is the index of idx_activity and idx_resource is the index of resource and "value" is an amount of the idx_resource resource consumed by the idx_activty activity

act_proc: "index" is the index of activity, "value" is duration of activity