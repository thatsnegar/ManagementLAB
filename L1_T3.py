import random
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt

# ******************************************************************************
# Constants
# ******************************************************************************

NUM_SERVERS = 3
SIM_TIME = 500000
ARRIVAL_TIME = 5.0
x = 5
SERVICE_TIMES = [x, x*2, x*3]
BUFFER_TYPE = 'shared'
BUFFER_SIZE = 5
TYPE1 = 1

STRATEGIES = ['random', 'round_robin', 'fastest']

# ******************************************************************************
# Server Data Structure
# ******************************************************************************
class Server:
    def __init__(self):
        self.idle = True

class Measure:
    def __init__(self):
        self.arr = 0
        self.dep = 0
        self.ut = 0.0
        self.oldT = 0.0
        self.delay = 0.0
        self.dropped = 0

class Client:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time

# ******************************************************************************
# Server selection strategy
# ******************************************************************************
def select_server(strategy):
    global rr_index
    if strategy == 'random':
        return random.randint(0, NUM_SERVERS - 1)
    elif strategy == 'round_robin':
        sid = rr_index
        rr_index = (rr_index + 1) % NUM_SERVERS
        return sid
    elif strategy == 'fastest':
        return np.argmin(SERVICE_TIMES)
    else:
        raise ValueError("Invalid strategy")

# ******************************************************************************
# Main simulation
# ******************************************************************************
def run_simulation(strategy):
    global rr_index
    servers = [Server() for _ in range(NUM_SERVERS)]
    measures = [Measure() for _ in range(NUM_SERVERS)]
    queues = [[] for _ in range(NUM_SERVERS)]
    rr_index = 0
    FES = PriorityQueue()
    current_time = 0.0

    def arrival(time):
        nonlocal current_time
        sid = select_server(strategy)
        m = measures[sid]
        q = queues[sid]

        inter_arrival = random.expovariate(1.0 / ARRIVAL_TIME)
        FES.put((time + inter_arrival, "arrival"))

        buffer_full = (
            (BUFFER_TYPE == 'per_server' and len(q) >= BUFFER_SIZE) or
            (BUFFER_TYPE == 'shared' and sum(len(queue) for queue in queues) >= BUFFER_SIZE)
        )

        if buffer_full:
            m.dropped += 1
            return

        m.arr += 1
        m.ut += len(q) * (time - m.oldT)
        m.oldT = time
        q.append(Client(time))

        if len(q) == 1 and servers[sid].idle:
            service_time = random.expovariate(1.0 / SERVICE_TIMES[sid])
            servers[sid].idle = False
            FES.put((time + service_time, f"departure_{sid}"))

    def departure(time, sid):
        m = measures[sid]
        q = queues[sid]

        m.dep += 1
        m.ut += len(q) * (time - m.oldT)
        m.oldT = time

        client = q.pop(0)
        m.delay += (time - client.arrival_time)

        if len(q) > 0:
            service_time = random.expovariate(1.0 / SERVICE_TIMES[sid])
            FES.put((time + service_time, f"departure_{sid}"))
        else:
            servers[sid].idle = True

    FES.put((0.0, "arrival"))

    while not FES.empty() and current_time < SIM_TIME:
        (current_time, event_type) = FES.get()

        if event_type == "arrival":
            arrival(current_time)
        elif event_type.startswith("departure_"):
            sid = int(event_type.split('_')[1])
            departure(current_time, sid)

    avg_delays = [m.delay / m.dep if m.dep > 0 else float('inf') for m in measures]

    print(f"\n--- Simulation results per server - {strategy} ---")
    print(f"\nPARAMETERS\nService times = {SERVICE_TIMES}\nBuffer size = {BUFFER_SIZE}")
    for i in range(NUM_SERVERS):
        m = measures[i]
        load = (1.0 / ARRIVAL_TIME) / (1.0 / SERVICE_TIMES[i])
        avg_users = m.ut / current_time if current_time > 0 else 0
        avg_delay = m.delay / m.dep if m.dep > 0 else float('inf')
        loss_prob = m.dropped / (m.arr + m.dropped) if (m.arr + m.dropped) > 0 else 0

        print(f"\nServer {i+1} (Service Time: {SERVICE_TIMES[i]}):")
        print(f"  Load: {load:.4f}")
        print(f"  Arrivals: {m.arr}, Departures: {m.dep}, Dropped: {m.dropped}")
        print(f"  Average Users: {avg_users:.4f}")
        print(f"  Average Delay: {avg_delay:.4f}")
        print(f"  Loss Probability: {loss_prob:.4f}")

    return avg_delays

# ******************************************************************************
# Run all strategies and collect results
# ******************************************************************************
strategy_delays = {}
for strat in STRATEGIES:
    random.seed(42)  # Reset seed for fair comparison
    strategy_delays[strat] = run_simulation(strat)

# ******************************************************************************
# Plotting: Average Delay per Server for Each Strategy
# ******************************************************************************

x = np.arange(NUM_SERVERS)  # server IDs
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
offset = -width
colors = ['#8cd3ff', '#008df9', '#2a33d4']

for idx, strat in enumerate(STRATEGIES):
    delays = strategy_delays[strat]
    ax.bar(x + offset, delays, width, label=strat.capitalize(), color=colors[idx])
    offset += width

ax.set_xlabel('Server ID')
ax.set_ylabel('Average Delay')
ax.set_xticks(x)
ax.set_xticklabels([str(i+1) for i in x])
ax.legend()

plt.tight_layout()
plt.show()