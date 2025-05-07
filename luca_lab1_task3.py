import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

SERVICE_RATES = [0.1, 0.2]  # Service rates for two servers (mu)
SIM_TIME = 500000     # Tempo totale di simulazione
BUFFER_SIZE = 10      # Dimensione massima del buffer condiviso

# Arrival rates da testare (diversi valori di lambda = 1/ARRIVAL)
arrival_times_to_test = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] # ARRIVAL times

TYPE1 = 1

# Policy: 'random', 'round_robin', 'fastest'
ASSIGN_POLICY = 'fastest'

# ==============================================================================
# Classi ausiliarie
# ==============================================================================

class Measure:
    def __init__(self):
        self.arr = 0
        self.dep = [0, 0]
        self.drop = 0
        self.ut = 0
        self.oldT = 0
        self.delay = [0, 0]

class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

class Server:
    def __init__(self, rate):
        self.queue = []
        self.busy = False
        self.rate = rate

# ==============================================================================
# Funzioni di simulazione
# ==============================================================================

def assign_server(servers, rr_counter, policy):
    if policy == 'random':
        return random.randint(0, len(servers) - 1)
    elif policy == 'round_robin':
        return rr_counter % len(servers)
    elif policy == 'fastest':
        return min(range(len(servers)), key=lambda i: servers[i].rate)
    else:
        return 0

def simulate(arrival_time_param):
    global users
    users = 0
    time = 0
    data = Measure()
    FES = PriorityQueue()
    servers = [Server(rate) for rate in SERVICE_RATES]
    buffer = []
    rr_counter = 0

    # Schedule primo arrivo
    FES.put((0, "arrival", rr_counter))

    while time < SIM_TIME:
        (time, event_type, sid) = FES.get()

        if event_type == "arrival":
            data.arr += 1
            data.ut += sum(len(s.queue) + s.busy for s in servers) * (time - data.oldT)
            data.oldT = time

            inter_arrival = random.expovariate(1.0/arrival_time_param)
            FES.put((time + inter_arrival, "arrival", rr_counter + 1))

            if len(buffer) < BUFFER_SIZE:
                client = Client(TYPE1, time)
                buffer.append(client)
                server_id = assign_server(servers, rr_counter, ASSIGN_POLICY)
                rr_counter += 1

                if not servers[server_id].busy:
                    servers[server_id].busy = True
                    service_time = random.expovariate(servers[server_id].rate)
                    servers[server_id].queue.append(client)
                    FES.put((time + service_time, "departure", server_id))
            else:
                data.drop += 1

        elif event_type == "departure":
            server = servers[sid]
            if server.queue:
                client = server.queue.pop(0)
                data.dep[sid] += 1
                data.delay[sid] += (time - client.arrival_time)

            data.ut += sum(len(s.queue) + s.busy for s in servers) * (time - data.oldT)
            data.oldT = time

            if buffer:
                next_client = buffer.pop(0)
                server.queue.append(next_client)
                service_time = random.expovariate(server.rate)
                FES.put((time + service_time, "departure", sid))
            else:
                server.busy = False

    avg_delay = sum(data.delay) / sum(data.dep) if sum(data.dep) > 0 else 0
    loss_probability = data.drop / (data.arr + data.drop) if (data.arr + data.drop) > 0 else 0
    return avg_delay, loss_probability

# ==============================================================================
# Main execution: variazione arrival rates e plot finale
# ==============================================================================

random.seed(42)

average_delays = []
loss_probabilities = []
loads = []

total_service_rate = sum(SERVICE_RATES)

for ARRIVAL in arrival_times_to_test:
    avg_delay, loss_probability = simulate(ARRIVAL)
    average_delays.append(avg_delay)
    loss_probabilities.append(loss_probability)
    loads.append((1.0 / ARRIVAL) / total_service_rate)

# ==============================================================================
# Plots
# ==============================================================================

plt.figure(figsize=(10,10))

# Plot Average Delay
plt.subplot(2, 1, 1)
plt.plot(loads, average_delays, marker='o')
plt.title(f'Average Delay vs Load ({ASSIGN_POLICY.capitalize()} Policy)')
plt.xlabel('Load (ρ)')
plt.ylabel('Average Delay')
plt.grid(True)

# Plot Loss Probability
plt.subplot(2, 1, 2)
plt.plot(loads, loss_probabilities, marker='x', color='red')
plt.title(f'Loss Probability vs Load ({ASSIGN_POLICY.capitalize()} Policy)')
plt.xlabel('Load (ρ)')
plt.ylabel('Packet Loss Probability')
plt.grid(True)

plt.tight_layout()
plt.show()