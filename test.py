# %%
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
SIM_TIME = 1000000
ARRIVAL_RATE = 1/10.0  # fixed average inter-arrival time
BUFFER_SIZE = 10       # shared total capacity (adjusted in separate mode)
SERVICE_RATES = [round(x, 2) for x in np.linspace(0.08, 0.25, 18)]
SEED = 42
TYPE1 = 1

# %%
class Measure:
    def __init__(self):
        self.arr = 0
        self.dep = 0
        self.ut = 0
        self.oldT = 0
        self.delay = 0
        self.bs1 = 0
        self.bs2 = 0
        self.loss = 0

class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

# %%
def simulate_buffer_config(service_rate, shared=True):
    global users, data, MM1, MM2, busy1, busy2
    users = 0
    data = Measure()
    MM1 = []
    MM2 = []
    time = 0
    busy1 = False
    busy2 = False
    start_time_1 = None
    start_time_2 = None

    FES = PriorityQueue()
    random.seed(SEED)

    def arrival(current_time):
        nonlocal start_time_1, start_time_2
        global users, busy1, busy2

        data.arr += 1
        data.ut += users * (current_time - data.oldT)
        data.oldT = current_time

        inter_arrival = random.expovariate(1.0 / ARRIVAL_RATE)
        FES.put((current_time + inter_arrival, "arrival"))

        if shared:
            if users < BUFFER_SIZE:
                users += 1
                MM1.append(Client(TYPE1, current_time))

                if not busy1:
                    s_time = random.expovariate(service_rate)
                    start_time_1 = current_time
                    FES.put((current_time + s_time, "departure1"))
                    busy1 = True
                elif not busy2:
                    s_time = random.expovariate(service_rate)
                    start_time_2 = current_time
                    FES.put((current_time + s_time, "departure2"))
                    busy2 = True
            else:
                data.loss += 1
        else:
            per_server_buffer = BUFFER_SIZE // 2
            target = 1 if random.random() < 0.5 else 2
            queue = MM1 if target == 1 else MM2
            if len(queue) < per_server_buffer:
                queue.append(Client(TYPE1, current_time))
                users += 1
                if target == 1 and not busy1:
                    s_time = random.expovariate(service_rate)
                    start_time_1 = current_time
                    FES.put((current_time + s_time, "departure1"))
                    busy1 = True
                elif target == 2 and not busy2:
                    s_time = random.expovariate(service_rate)
                    start_time_2 = current_time
                    FES.put((current_time + s_time, "departure2"))
                    busy2 = True
            else:
                data.loss += 1

    def departure(current_time, server_id):
        nonlocal start_time_1, start_time_2
        global users, busy1, busy2

        data.dep += 1
        data.ut += users * (current_time - data.oldT)
        data.oldT = current_time

        if shared:
            if MM1:
                client = MM1.pop(0)
                data.delay += (current_time - client.arrival_time)
                users -= 1
                if MM1:
                    s_time = random.expovariate(service_rate)
                    if server_id == 1:
                        data.bs1 += current_time - start_time_1
                        start_time_1 = current_time
                        FES.put((current_time + s_time, "departure1"))
                    else:
                        data.bs2 += current_time - start_time_2
                        start_time_2 = current_time
                        FES.put((current_time + s_time, "departure2"))
                else:
                    if server_id == 1:
                        data.bs1 += current_time - start_time_1
                        busy1 = False
                    else:
                        data.bs2 += current_time - start_time_2
                        busy2 = False
        else:
            queue = MM1 if server_id == 1 else MM2
            if queue:
                client = queue.pop(0)
                data.delay += (current_time - client.arrival_time)
                users -= 1
                if queue:
                    s_time = random.expovariate(service_rate)
                    if server_id == 1:
                        data.bs1 += current_time - start_time_1
                        start_time_1 = current_time
                        FES.put((current_time + s_time, "departure1"))
                    else:
                        data.bs2 += current_time - start_time_2
                        start_time_2 = current_time
                        FES.put((current_time + s_time, "departure2"))
                else:
                    if server_id == 1:
                        data.bs1 += current_time - start_time_1
                        busy1 = False
                    else:
                        data.bs2 += current_time - start_time_2
                        busy2 = False

    FES.put((0, "arrival"))
    while time < SIM_TIME:
        (time, event_type) = FES.get()
        if event_type == "arrival":
            arrival(time)
        elif event_type == "departure1":
            departure(time, 1)
        elif event_type == "departure2":
            departure(time, 2)

    delay = data.delay / data.dep if data.dep > 0 else 0
    avg_users = data.ut / time
    utilization = (data.bs1 + data.bs2) / (time * 2)
    loss_rate = data.loss / data.arr if data.arr > 0 else 0

    return delay, avg_users, utilization, loss_rate

# %%
# Run simulation for each service rate (Task 2.c)
results = []
for servicetime in SERVICE_RATES:
    shared = simulate_buffer_config(1 / servicetime, shared=True)
    separate = simulate_buffer_config(1 / servicetime, shared=False)
    results.append({
        'Service Rate': servicetime,
        'Type': 'Shared',
        'Avg Delay': shared[0],
        'Avg Users': shared[1],
        'Utilization': shared[2],
        'Loss Rate': shared[3]
    })
    results.append({
        'Service Rate': servicetime,
        'Type': 'Separate',
        'Avg Delay': separate[0],
        'Avg Users': separate[1],
        'Utilization': separate[2],
        'Loss Rate': separate[3]
    })

# %%
# Create DataFrame
df_results = pd.DataFrame(results)
print(df_results)

# %%
# Plotting
metrics = ['Avg Delay', 'Avg Users', 'Utilization', 'Loss Rate']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for t in ['Shared', 'Separate']:
        subset = df_results[df_results['Type'] == t]
        plt.plot(subset['Service Rate'], subset[metric], marker='o', label=t)
    plt.title(f'{metric} vs Service Rate (Fixed λ = {1/ARRIVAL_RATE})')
    plt.xlabel('Service Rate (μ)')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
