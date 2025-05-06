import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

# ****************************************************************************
# Constants
# ****************************************************************************

ARRIVAL = 5.0  # Fixed average inter-arrival time
SIM_TIME = 500000  # Total simulation time
SERVICE_VALUES = [i for i in range(5, 41, 2)]  # Average service times: 5.0 to 40.0 in steps of 2


# ****************************************************************************
# Classes
# ****************************************************************************
class Measure:
    def __init__(self, arr=0, dep=0, ut=0, oldT=0, delay=0):
        self.arr = arr
        self.dep = dep
        self.ut = ut
        self.oldT = oldT
        self.delay = delay

class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

# ****************************************************************************
# Functions
# ****************************************************************************
def arrival(time, FES, queue, data, SERVICE):
    data.arr += 1
    data.ut += len(queue) * (time - data.oldT)
    data.oldT = time

    inter_arrival = random.expovariate(1.0 / ARRIVAL)
    FES.put((time + inter_arrival, "arrival"))

    client = Client(1, time)
    queue.append(client)

    if len(queue) == 1:
        service_time = random.expovariate(1.0 / SERVICE)
        FES.put((time + service_time, "departure"))

def departure(time, FES, queue, data, SERVICE):
    data.dep += 1
    data.ut += len(queue) * (time - data.oldT)
    data.oldT = time

    client = queue.pop(0)
    data.delay += (time - client.arrival_time)

    if len(queue) > 0:
        service_time = random.expovariate(1.0 / SERVICE)
        FES.put((time + service_time, "departure"))

# ****************************************************************************
# Main Simulation Loop
# ****************************************************************************
results = []

for SERVICE in SERVICE_VALUES:
    random.seed(42)
    data = Measure()
    time = 0
    FES = PriorityQueue()
    MM1 = []

    FES.put((0, "arrival"))

    while time < SIM_TIME:
        (time, event_type) = FES.get()

        if event_type == "arrival":
            arrival(time, FES, MM1, data, SERVICE)
        elif event_type == "departure":
            departure(time, FES, MM1, data, SERVICE)

    results.append({
        'Service Time': SERVICE,
        'Load': SERVICE / ARRIVAL,
        'Arrival Rate': data.arr / time,
        'Departure Rate': data.dep / time,
        'Average Users': data.ut / time,
        'Average Delay': data.delay / data.dep,
        'Queue Size at End': len(MM1)
    })

# Plotting the results
loads = [r['Load'] for r in results]
avg_delays = [r['Average Delay'] for r in results]
avg_users = [r['Average Users'] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(loads, avg_delays, marker='o')
plt.xlabel("System Load (ρ = SERVICE / ARRIVAL)")
plt.ylabel("Average Delay")
plt.title("Average Delay vs Load")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(loads, avg_users, marker='o')
plt.xlabel("System Load (ρ = SERVICE / ARRIVAL)")
plt.ylabel("Average Number of Users")
plt.title("Average Users vs Load")
plt.grid(True)
plt.tight_layout()
plt.show()
