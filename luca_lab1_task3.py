import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

# ==============================================================================
# params
# ==============================================================================
ARRIVAL = 5.0              # Tempo medio tra arrivi (=> λ = 1/ARRIVAL)
SERVICE_RATES = [1/10.0, 1/5.0, 1/7.0]  # Tassi di servizio dei server (μ)
SIM_TIME = 500000


class Measure:
    def __init__(self):
        self.arr = 0
        self.dep = [0, 0]
        self.ut = 0
        self.oldT = 0
        self.delay = [0, 0]
        self.busy_time = [0, 0]

class Client:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time

class Server:
    def __init__(self, rate):
        self.queue = []
        self.busy = False
        self.rate = rate
        self.last_start_time = 0

# ==============================================================================
# functions
# ==============================================================================
def assign_server(servers, time, policy, rr_index):
    if policy == 'random':
        return random.randint(0, len(servers) - 1)
    elif policy == 'round_robin':
        return rr_index % len(servers)
    elif policy == 'fastest':
        return min(range(len(servers)), key=lambda i: servers[i].rate)
    return 0

def arrival(time, FES, servers, data, policy, rr_index):
    data.arr += 1
    data.ut += sum(len(s.queue) + s.busy for s in servers) * (time - data.oldT)
    data.oldT = time

    # Schedule next arrival
    inter_arrival = random.expovariate(1.0 / ARRIVAL)
    FES.put((time + inter_arrival, "arrival", rr_index + 1))

    # Assign to server
    sid = assign_server(servers, time, policy, rr_index)
    client = Client(time)
    servers[sid].queue.append(client)

    if not servers[sid].busy:
        service_time = random.expovariate(servers[sid].rate)
        servers[sid].busy = True
        servers[sid].last_start_time = time
        FES.put((time + service_time, "departure", sid))

def departure(time, FES, servers, data, sid):
    server = servers[sid]
    client = server.queue.pop(0)

    data.dep[sid] += 1
    data.delay[sid] += (time - client.arrival_time)
    data.ut += sum(len(s.queue) + s.busy for s in servers) * (time - data.oldT)
    data.oldT = time

    # Busy time per server
    data.busy_time[sid] += (time - server.last_start_time)

    if server.queue:
        service_time = random.expovariate(server.rate)
        server.last_start_time = time
        FES.put((time + service_time, "departure", sid))
    else:
        server.busy = False

# ==============================================================================
# simulation
# ==============================================================================
def simulate(policy):
    random.seed(900)
    data = Measure()
    servers = [Server(rate) for rate in SERVICE_RATES]
    FES = PriorityQueue()
    FES.put((0, "arrival", 0))
    time = 0

    while time < SIM_TIME:
        time, event_type, sid = FES.get()
        if event_type == "arrival":
            arrival(time, FES, servers, data, policy, sid)
        elif event_type == "departure":
            departure(time, FES, servers, data, sid)

    return data

# ==============================================================================
# for each policy
# ==============================================================================
policies = ['random', 'round_robin', 'fastest']
results = {}

for policy in policies:
    print(f"Running simulation with policy: {policy}")
    results[policy] = simulate(policy)

# ==============================================================================
# plots
# ==============================================================================
x = policies
delays = [sum(results[p].delay) / sum(results[p].dep) for p in policies]
busy = [[bt / SIM_TIME for bt in results[p].busy_time] for p in policies]

plt.figure(figsize=(8, 5))
plt.bar(x, delays)
plt.ylabel("Average Delay per Packet")
plt.title("Average Delay per Policy")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
for i in range(len(SERVICE_RATES)):
    plt.plot(x, [b[i] for b in busy], label=f"Server {i+1}")
plt.ylabel("Server Utilization (Busy Time / Total Time)")
plt.title("Server Busy Time per Policy")
plt.legend()
plt.grid(True)
plt.show()