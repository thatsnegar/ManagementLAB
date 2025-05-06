# %%
# The purpose of task is to evaluate how changing the buffer size in multi-server queuing system 
# effects system performance 

# -- if the buffer is too small the system starts dropping packets 
# -- become less efficient 
# -- may cause delay

# Also extended to evaluate performance across varying arrival and service rates

# %%
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
SIM_TIME = 500000
TYPE1 = 1
SEED = 42
BUFFER_SIZE = 6  # Fixed buffer size for both shared and separate buffer scenarios

# Arrival and service rate combinations to test
ARRIVAL_RATES = [8.0, 10.0, 12.0]
SERVICE_RATES = [5.0, 7.0, 9.0]

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

# %%
class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

# %%
def simulate(arrival_rate, service_rate, shared=True):
    global users, data, MM1, MM2, busy1, busy2
    users = 0
    data = Measure()
    MM1 = []
    MM2 = []
    time = 0
    busy1 = False
    busy2 = False

    FES = PriorityQueue()
    random.seed(SEED)

    def arrival(current_time):
        global users, busy1, busy2
        data.arr += 1
        data.ut += users * (current_time - data.oldT)
        data.oldT = current_time

        inter_arrival = random.expovariate(1.0 / arrival_rate)
        FES.put((current_time + inter_arrival, "arrival"))

        if shared:
            if users < BUFFER_SIZE:
                users += 1
                client = Client(TYPE1, current_time)
                MM1.append(client)

                if not busy1:
                    service_time = random.expovariate(1.0 / service_rate)
                    data.bs1 += service_time
                    FES.put((current_time + service_time, "departure1"))
                    busy1 = True
                elif not busy2:
                    service_time = random.expovariate(1.0 / service_rate)
                    data.bs2 += service_time
                    FES.put((current_time + service_time, "departure2"))
                    busy2 = True
            else:
                data.loss += 1
        else:
            target = 1 if random.random() < 0.5 else 2
            buffer_limit = BUFFER_SIZE
            if target == 1:
                if len(MM1) < buffer_limit:
                    MM1.append(Client(TYPE1, current_time))
                    users += 1
                    if not busy1:
                        service_time = random.expovariate(1.0 / service_rate)
                        data.bs1 += service_time
                        FES.put((current_time + service_time, "departure1"))
                        busy1 = True
                else:
                    data.loss += 1
            else:
                if len(MM2) < buffer_limit:
                    MM2.append(Client(TYPE1, current_time))
                    users += 1
                    if not busy2:
                        service_time = random.expovariate(1.0 / service_rate)
                        data.bs2 += service_time
                        FES.put((current_time + service_time, "departure2"))
                        busy2 = True
                else:
                    data.loss += 1

    def departure(current_time, server_id):
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
                    service_time = random.expovariate(1.0 / service_rate)
                    if server_id == 1:
                        data.bs1 += service_time
                        FES.put((current_time + service_time, "departure1"))
                    else:
                        data.bs2 += service_time
                        FES.put((current_time + service_time, "departure2"))
                else:
                    if server_id == 1:
                        busy1 = False
                    else:
                        busy2 = False
        else:
            target_queue = MM1 if server_id == 1 else MM2
            if target_queue:
                client = target_queue.pop(0)
                data.delay += (current_time - client.arrival_time)
                users -= 1
                if target_queue:
                    service_time = random.expovariate(1.0 / service_rate)
                    if server_id == 1:
                        data.bs1 += service_time
                        FES.put((current_time + service_time, "departure1"))
                    else:
                        data.bs2 += service_time
                        FES.put((current_time + service_time, "departure2"))
                else:
                    if server_id == 1:
                        busy1 = False
                    else:
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
# Run experiments over different arrival/service rates
results = []

for arrival in ARRIVAL_RATES:
    for service in SERVICE_RATES:
        shared_metrics = simulate(arrival_rate=arrival, service_rate=service, shared=True)
        separate_metrics = simulate(arrival_rate=arrival, service_rate=service, shared=False)
        results.append({
            'Arrival': arrival,
            'Service': service,
            'Type': 'Shared',
            'Avg Delay': shared_metrics[0],
            'Avg Users': shared_metrics[1],
            'Utilization': shared_metrics[2],
            'Loss Rate': shared_metrics[3]
        })
        results.append({
            'Arrival': arrival,
            'Service': service,
            'Type': 'Separate',
            'Avg Delay': separate_metrics[0],
            'Avg Users': separate_metrics[1],
            'Utilization': separate_metrics[2],
            'Loss Rate': separate_metrics[3]
        })

# %%
# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# %%
# Plotting one metric at a time without seaborn
metrics = ['Avg Delay', 'Avg Users', 'Utilization', 'Loss Rate']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for service in SERVICE_RATES:
        shared_vals = results_df[(results_df['Type'] == 'Shared') & (results_df['Service'] == service)][metric].values
        separate_vals = results_df[(results_df['Type'] == 'Separate') & (results_df['Service'] == service)][metric].values
        plt.plot(ARRIVAL_RATES, shared_vals, marker='o', label=f'Shared - Service {service}')
        plt.plot(ARRIVAL_RATES, separate_vals, marker='s', linestyle='--', label=f'Separate - Service {service}')
    plt.title(f'{metric} vs Arrival Rate at Different Service Rates')
    plt.xlabel('Arrival Rate')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
