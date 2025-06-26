#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue, PriorityQueue

# ******************************************************************************
# Constants
# ******************************************************************************

ARRIVAL = 3.0  # ARRIVAL is the average inter-arrival time; arrival rate = 1/ARRIVAL = 1/3
TYPE1 = 1 
SIM_TIME = 200000

# ******************************************************************************
# To take the measurements
# ******************************************************************************
class Measure:
    def __init__(self, Narr, Ndep, NAveraegUser, OldTimeEvent, AverageDelay):
        self.arr = Narr
        self.dep = Ndep
        self.ut = NAveraegUser
        self.oldT = OldTimeEvent
        self.delay = AverageDelay

# ******************************************************************************
# Client
# ******************************************************************************
class Client:
    def __init__(self, type, arrival_time):
        self.type = type
        self.arrival_time = arrival_time

# ******************************************************************************
# Service time generation functions
# ******************************************************************************
def generate_service_time(distribution, service_rate):
    """Generate service time based on distribution type"""
    mean_service = 1.0 / service_rate
    
    if distribution == "exponential":
        return random.expovariate(service_rate)
    
    elif distribution == "uniform":
        # Uniform(0, 2/mu) has mean 1/mu
        return random.uniform(0, 2 * mean_service)
    
    elif distribution == "deterministic":
        return mean_service
    
    elif distribution == "hyperexponential":
        # Mixture: 0.8 * Exp(2*mu) + 0.2 * Exp(0.4*mu)
        # This gives mean = 0.8*(1/2mu) + 0.2*(1/0.4mu) = 0.4/mu + 0.5/mu = 0.9/mu
        # To get mean = 1/mu, we adjust: 0.8 * Exp(1.8*mu) + 0.2 * Exp(0.36*mu)
        if random.random() < 0.8:
            return random.expovariate(1.8 * service_rate)
        else:
            return random.expovariate(0.36 * service_rate)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

# ******************************************************************************
# Simulation functions
# ******************************************************************************

def arrival(time, FES, queue, users, data):
    """Handle arrival event"""
    # cumulate statistics
    data.arr += 1
    data.ut += users[0] * (time - data.oldT)
    data.oldT = time

    # sample the time until the next event
    inter_arrival = random.expovariate(1.0/ARRIVAL)
    
    # schedule the next arrival
    FES.put((time + inter_arrival, "arrival"))

    users[0] += 1
    
    # create a record for the client
    client = Client(TYPE1, time)

    # insert the record in the queue
    queue.append(client)

    return users[0] == 1  # return True if server was idle

def departure(time, FES, queue, users, data, distribution, service_rate):
    """Handle departure event"""
    # cumulate statistics
    data.dep += 1
    data.ut += users[0] * (time - data.oldT)
    data.oldT = time
    
    # get the first element from the queue
    client = queue.pop(0)
    
    # calculate delay
    data.delay += (time - client.arrival_time)
    users[0] -= 1
    
    return users[0] > 0  # return True if there are more clients

def simulate_mm1_distribution(distribution, service_rates):
    """Simulate M/G/1 queue with different service time distributions"""
    results = {
        'service_rates': [],
        'avg_delay': [],
        'avg_users': [],
        'utilization': []
    }
    
    for service_rate in service_rates:
        random.seed(42)  # For reproducible results
        
        data = Measure(0, 0, 0, 0, 0)
        users = [0]  # Use list to make it mutable
        queue = []
        
        # the simulation time 
        time = 0
        
        # the list of events in the form: (time, type)
        FES = PriorityQueue()
        
        # schedule the first arrival at t=0
        FES.put((0, "arrival"))
        
        # simulate until the simulated time reaches a constant
        while time < SIM_TIME:
            (time, event_type) = FES.get()
            
            if event_type == "arrival":
                server_was_idle = arrival(time, FES, queue, users, data)
                
                # if the server was idle, start the service
                if server_was_idle:
                    service_time = generate_service_time(distribution, service_rate)
                    FES.put((time + service_time, "departure"))
            
            elif event_type == "departure":
                more_clients = departure(time, FES, queue, users, data, distribution, service_rate)
                
                # if there are more clients, start next service
                if more_clients:
                    service_time = generate_service_time(distribution, service_rate)
                    FES.put((time + service_time, "departure"))
        
        # Calculate metrics
        if data.dep > 0:
            avg_delay = data.delay / data.dep
            avg_users = data.ut / time
            utilization = (1.0/ARRIVAL) / service_rate  # lambda / mu
            
            results['service_rates'].append(service_rate)
            results['avg_delay'].append(avg_delay)
            results['avg_users'].append(avg_users)
            results['utilization'].append(utilization)
    
    return results

# ******************************************************************************
# Main simulation and plotting
# ******************************************************************************

def main():
    # Service rates to test (same as Task 1.a)
    service_rates = list(range(5, 40, 2))  # 5, 7, 9, ..., 39
    
    # Distributions to test
    distributions = ["exponential", "uniform", "deterministic", "hyperexponential"]
    
    # Store all results
    all_results = {}
    
    print("Running simulations for different service time distributions...")
    
    for dist in distributions:
        print(f"Simulating {dist} distribution...")
        all_results[dist] = simulate_mm1_distribution(dist, service_rates)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Task 4: Service Time Distribution Comparison (M/G/1)', fontsize=16)
    
    # Plot 1: Average Delay vs Service Rate
    ax1 = axes[0, 0]
    for dist in distributions:
        ax1.plot(all_results[dist]['service_rates'], all_results[dist]['avg_delay'], 
                marker='o', label=dist.capitalize(), linewidth=2, markersize=4)
    ax1.set_xlabel('Service Rate μ (packets/second)')
    ax1.set_ylabel('Average Delay (s)')
    ax1.set_title('Average Delay vs Service Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Users vs Service Rate
    ax2 = axes[0, 1]
    for dist in distributions:
        ax2.plot(all_results[dist]['service_rates'], all_results[dist]['avg_users'], 
                marker='s', label=dist.capitalize(), linewidth=2, markersize=4)
    ax2.set_xlabel('Service Rate μ (packets/second)')
    ax2.set_ylabel('Average Number of Users')
    ax2.set_title('Average Users vs Service Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Utilization vs Service Rate (should be same for all)
    ax3 = axes[1, 0]
    for dist in distributions:
        ax3.plot(all_results[dist]['service_rates'], all_results[dist]['utilization'], 
                marker='^', label=dist.capitalize(), linewidth=2, markersize=4)
    ax3.set_xlabel('Service Rate μ (packets/second)')
    ax3.set_ylabel('Utilization ρ')
    ax3.set_title('Utilization vs Service Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance comparison at μ=15
    ax4 = axes[1, 1]
    mu_15_index = service_rates.index(15) if 15 in service_rates else len(service_rates)//2
    
    delays_at_mu15 = [all_results[dist]['avg_delay'][mu_15_index] for dist in distributions]
    colors = ['blue', 'orange', 'green', 'red']
    
    bars = ax4.bar(distributions, delays_at_mu15, color=colors, alpha=0.7)
    ax4.set_ylabel('Average Delay (s)')
    ax4.set_title(f'Delay Comparison at μ = {service_rates[mu_15_index]}')
    ax4.set_xticklabels([d.capitalize() for d in distributions], rotation=45)
    
    # Add value labels on bars
    for bar, delay in zip(bars, delays_at_mu15):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{delay:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('task4_service_distributions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("TASK 4 - SERVICE TIME DISTRIBUTION ANALYSIS SUMMARY")
    print("="*80)
    
    mu_test = 15  # Example service rate
    if mu_test in service_rates:
        idx = service_rates.index(mu_test)
        print(f"\nPerformance at μ = {mu_test} packets/second (ρ = {1/ARRIVAL/mu_test:.3f}):")
        print("-" * 60)
        print(f"{'Distribution':<15} {'Avg Delay (s)':<12} {'Avg Users':<12} {'Relative':<10}")
        print("-" * 60)
        
        exp_delay = all_results['exponential']['avg_delay'][idx]
        for dist in distributions:
            delay = all_results[dist]['avg_delay'][idx]
            users = all_results[dist]['avg_users'][idx]
            relative = delay / exp_delay
            print(f"{dist.capitalize():<15} {delay:<12.4f} {users:<12.3f} {relative:<10.3f}")
    
    # Calculate coefficient of variation effects
    print(f"\n\nCoefficient of Variation Analysis:")
    print("-" * 40)
    cv_values = {
        'deterministic': 0.0,
        'uniform': 1/3,
        'exponential': 1.0,
        'hyperexponential': 2.25  # Approximate for our mixture
    }
    
    for dist in distributions:
        print(f"{dist.capitalize():<15}: C²ₛ = {cv_values[dist]:.2f}")

if __name__ == "__main__":
    main()