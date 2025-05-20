# M/M/1 Queue Simulation with Dynamic Warm-Up Detection and Departure Delay Plotting
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt

# ******************************************************************************
# Constants
# ******************************************************************************
SERVICE = 10.0
ARRIVAL = 5.0
SIM_TIME = 500000
LOG_INTERVAL = 1000
epsilon = 0.0005
detection_window = 20

TYPE1 = 1
arrivals = 0
users = 0

# ******************************************************************************
# Classes
# ******************************************************************************
class Measure:
    def __init__(self, Narr, Ndep, NAveraegUser, OldTimeEvent, AverageDelay):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay

class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

class Server:
    def __init__(self):
        self.idle = True

# ******************************************************************************
# Event Functions
# ******************************************************************************
def arrival(time, FES, queue):
    global users
    data.arr += 1
    data.ut += users * (time - data.oldT)
    data.oldT = time
    inter_arrival = random.expovariate(1.0 / ARRIVAL)
    FES.put((time + inter_arrival, "arrival"))
    users += 1
    client = Client(TYPE1, time)
    queue.append(client)
    if users == 1:
        service_time = random.expovariate(1.0 / SERVICE)
        FES.put((time + service_time, "departure"))

def departure(time, FES, queue):
    global users, avg_delay_prev, steady_detected, steady_time, recent_changes

    data.dep += 1
    data.ut += users * (time - data.oldT)
    data.oldT = time
    client = queue.pop(0)
    users -= 1

    # measure delay here  
    delay = time - client.arrival_time
    data.delay += delay

    #  compute the current avg delay 
    avg_delay_curr = data.delay / data.dep
    departure_log.append((time, avg_delay_curr))  # Track for plotting

    # once we have more than 1 departure 
    # compute how much the avg delay changed 
    # stor delay evolution for plotting 
    if data.dep > 1:
        delta = abs(avg_delay_curr - avg_delay_prev)
        relative_change = delta / avg_delay_prev if avg_delay_prev > 0 else 1
        delay_evolution.append((time, avg_delay_curr))

        # check if the change is less than epsilon
        if not steady_detected:
            recent_changes.append(relative_change)
            if len(recent_changes) > detection_window:
                recent_changes.pop(0)
                if all(change < epsilon for change in recent_changes):
                    steady_detected = True
                    steady_time = time
    avg_delay_prev = avg_delay_curr

    if users > 0:
        service_time = random.expovariate(1.0 / SERVICE)
        FES.put((time + service_time, "departure"))

# ******************************************************************************
# Main Simulation
# ******************************************************************************
random.seed(42)
data = Measure(0, 0, 0, 0, 0)
time = 0
FES = PriorityQueue()
MM1 = []
FES.put((0, "arrival"))

avg_delay_prev = 0
delay_evolution = []
departure_log = []  # New: log of (departure time, avg delay)
recent_changes = []
steady_detected = False
steady_time = None
next_log_time = LOG_INTERVAL
logs = []

while time < SIM_TIME:
    (time, event_type) = FES.get()
    if event_type == "arrival":
        arrival(time, FES, MM1)
    elif event_type == "departure":
        departure(time, FES, MM1)

    if time >= next_log_time:
        avg_users = data.ut / time
        avg_delay = data.delay / data.dep if data.dep > 0 else 0
        logs.append((time, avg_users, avg_delay))
        next_log_time += LOG_INTERVAL

# ******************************************************************************
# Output and Plotting
# ******************************************************************************
if steady_detected:
    print(f"Steady state detected at time: {steady_time}")
else:
    print("Steady state not detected within simulation time.")

# Plot delay evolution (for steady state detection)
times = [t for (t, d) in delay_evolution]
delays = [d for (t, d) in delay_evolution]

plt.figure()
plt.plot(times, delays, label='Avg Delay (Steady-State Detection)')
if steady_detected:
    plt.axvline(x=steady_time, color='r', linestyle='--', label='Steady State Detected')
plt.xlabel("Time")
plt.ylabel("Average Delay")
plt.title("Average Delay Over Time (for Steady State Detection)")
plt.grid(True)
plt.legend()
plt.show()

# Plot departure time vs average delay
dep_times = [t for (t, d) in departure_log]
dep_delays = [d for (t, d) in departure_log]

plt.figure()
plt.plot(dep_times, dep_delays, label='Avg Delay at Each Departure')
plt.xlabel("Departure Time")
plt.ylabel("Average Delay")
plt.title("Average Delay vs Departure Time")
plt.grid(True)
plt.legend()
plt.show()
