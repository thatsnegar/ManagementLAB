import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

# ****************************************************************************
# Simulation Parameters
# ****************************************************************************
SIM_TIME = 100000
NUM_BS = 3
HAPS_CAPACITY = 100
HAPS_SERVICE_RATE = 1/5.0
BS_SERVICE_RATE = 1/5.0
BS_BUFFER_CAPACITY = 100
USE_HAPS = True

hourly_arrival_rates = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

# ****************************************************************************
# Data Collection
# ****************************************************************************
class Measure:
    def __init__(self):
        self.total_arrivals = 0
        self.bs_served = [0]*NUM_BS
        self.haps_served = 0
        self.bs_busy_time = [0.0]*NUM_BS
        self.haps_busy_time = 0.0
        self.total_delay = []
        self.haps_delay = []
        self.arrivals_per_hour = [0]*12
        self.haps_served_per_hour = [0]*12
        self.bs_served_per_hour = [0]*12
        self.bs_drops = [0]*NUM_BS

metrics = Measure()

# ****************************************************************************
# Event System
# ****************************************************************************
class Packet:
    def __init__(self, arrival_time):
        self.arrival_time = arrival_time

FES = PriorityQueue()
time = 0.0

haps_buffer = []
bs_buffers = [[] for _ in range(NUM_BS)]
bs_busy = [False]*NUM_BS
haps_busy = False

# ****************************************************************************
# Event Handlers
# ****************************************************************************
def schedule_event(event_time, event_type, payload=None):
    FES.put((event_time, event_type, payload))

def handle_arrival():
    global time
    hour = int(time // 3600) % 12
    lam = hourly_arrival_rates[hour]
    iat = random.expovariate(lam)
    schedule_event(time + iat, "arrival")

    metrics.total_arrivals += 1
    metrics.arrivals_per_hour[hour] += 1
    packet = Packet(time)

    if USE_HAPS and len(haps_buffer) < HAPS_CAPACITY and not haps_busy:
        haps_buffer.append(packet)
        start_haps_service(packet)
    else:
        min_index = min(range(NUM_BS), key=lambda i: len(bs_buffers[i]))
        if len(bs_buffers[min_index]) < BS_BUFFER_CAPACITY:
            bs_buffers[min_index].append(packet)
            if not bs_busy[min_index]:
                start_bs_service(min_index, packet)
        else:
            metrics.bs_drops[min_index] += 1

def start_haps_service(packet):
    global haps_busy
    haps_busy = True
    service_time = random.expovariate(HAPS_SERVICE_RATE)
    schedule_event(time + service_time, "haps_departure", packet)
    metrics.haps_busy_time += service_time

def handle_haps_departure(packet):
    global haps_busy
    haps_busy = False
    delay = time - packet.arrival_time
    metrics.total_delay.append(delay)
    metrics.haps_delay.append(delay)
    metrics.haps_served += 1
    hour = int(time // 3600) % 12
    metrics.haps_served_per_hour[hour] += 1
    if haps_buffer:
        next_packet = haps_buffer.pop(0)
        start_haps_service(next_packet)

def start_bs_service(index, packet):
    global bs_busy
    bs_busy[index] = True
    service_time = random.expovariate(BS_SERVICE_RATE)
    schedule_event(time + service_time, f"bs_departure_{index}", packet)
    metrics.bs_busy_time[index] += service_time

def handle_bs_departure(index, packet):
    global bs_busy
    bs_busy[index] = False
    delay = time - packet.arrival_time
    metrics.total_delay.append(delay)
    metrics.bs_served[index] += 1
    hour = int(time // 3600) % 12
    metrics.bs_served_per_hour[hour] += 1
    if bs_buffers[index]:
        next_packet = bs_buffers[index].pop(0)
        start_bs_service(index, next_packet)

# ****************************************************************************
# Main Simulation
# ****************************************************************************
def run_simulation():
    global time
    random.seed(42)
    schedule_event(0, "arrival")

    while not FES.empty() and time < SIM_TIME:
        event_time, event_type, payload = FES.get()
        time = event_time

        if event_type == "arrival":
            handle_arrival()
        elif event_type == "haps_departure":
            handle_haps_departure(payload)
        elif event_type.startswith("bs_departure_"):
            index = int(event_type.split("_")[-1])
            handle_bs_departure(index, payload)

    # Results
    scenario = "Scenario B (with HAPS)" if USE_HAPS else "Scenario A (BS only)"
    print(f"\n--- Simulation Results: {scenario} ---")
    print(f"Total Arrivals: {metrics.total_arrivals}")
    print(f"HAPS Served: {metrics.haps_served}")
    print(f"HAPS Busy Time: {metrics.haps_busy_time:.2f} seconds")
    for i in range(NUM_BS):
        print(f"BS[{i}] Served: {metrics.bs_served[i]}")
        print(f"BS[{i}] Busy Time: {metrics.bs_busy_time[i]:.2f} seconds")
        print(f"BS[{i}] Packet Drops: {metrics.bs_drops[i]}")
    avg_delay = sum(metrics.total_delay)/len(metrics.total_delay)
    print(f"Average Total Delay: {avg_delay:.2f} seconds")
    if metrics.haps_delay:
        avg_haps_delay = sum(metrics.haps_delay)/len(metrics.haps_delay)
        print(f"Average HAPS Delay: {avg_haps_delay:.2f} seconds")

    # Plotting: Performance Evaluation
    hours = list(range(8, 20))
    plt.figure(figsize=(12, 6))
    plt.plot(hours, metrics.arrivals_per_hour, label='Total Arrivals', linestyle='--', marker='o')
    if USE_HAPS:
        plt.plot(hours, metrics.haps_served_per_hour, label='HAPS Served', marker='x')
    plt.plot(hours, metrics.bs_served_per_hour, label='BSs Served', marker='s')
    plt.xlabel('Hour of Day')
    plt.ylabel('Packets')
    plt.title(f'Traffic Handling by Hour - {scenario}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Busy Time per BS
    plt.figure(figsize=(8, 5))
    plt.bar(range(NUM_BS), metrics.bs_busy_time, tick_label=[f"BS{i}" for i in range(NUM_BS)])
    plt.ylabel('Total Busy Time (s)')
    plt.title('Average Busy Time per Terrestrial BS')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Plot: Delay Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(metrics.total_delay, bins=50, alpha=0.7)
    plt.xlabel('Delay (s)')
    plt.ylabel('Packet Count')
    plt.title('Distribution of Total Queuing and Waiting Delays')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Packet Drops per BS
    plt.figure(figsize=(8, 5))
    plt.bar(range(NUM_BS), metrics.bs_drops, tick_label=[f"BS{i}" for i in range(NUM_BS)], color='red')
    plt.ylabel('Dropped Packets')
    plt.title('Packet Drops per BS due to Buffer Overflow')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Fraction of traffic handled by HAPS
    if USE_HAPS:
        fraction_haps = metrics.haps_served / metrics.total_arrivals
        print(f"Fraction of traffic handled by HAPS: {fraction_haps:.2%}")

if __name__ == "__main__":
    run_simulation()
