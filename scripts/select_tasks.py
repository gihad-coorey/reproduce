import json

tasks = [
	"libero_spatial_task_0",
	"libero_spatial_task_1",
	"libero_object_task_0",
	"libero_object_task_1",
	"libero_goal_task_0"
]

with open("configs/tasks.json","w") as f:
	json.dump(tasks,f,indent=2)

print("Saved task list.")
