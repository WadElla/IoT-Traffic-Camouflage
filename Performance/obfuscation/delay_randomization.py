from scapy.all import rdpcap, wrpcap
import random
import time
import psutil
import os
def add_delay_to_traffic(input_pcap, output_pcap, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    delayed_packets = []
    
    for packet in packets:
        random_delay = random.uniform(min_delay, max_delay)
        time.sleep(random_delay)  # Introduce actual delay
        packet.time = time.time()  # Set the timestamp to the current time after the delay
        delayed_packets.append(packet)
            
    wrpcap(output_pcap, delayed_packets)

if __name__ == "__main__":
    input_pcap = 'st_dataset/light.pcap'  
    output_pcap = 'st_dataset/st_delay/light_st_redel.pcap'
    min_delay = 0.01  # Minimum delay in seconds (10ms)
    max_delay = 0.2   # Maximum delay in seconds (100ms)

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Add delay to the traffic
    print("Adding delay to traffic...")
    start_time = time.time()
    add_delay_to_traffic(input_pcap, output_pcap, min_delay, max_delay)
    end_time = time.time()
    delay_time = end_time - start_time
    print(f"Delayed packets saved to: {output_pcap}")

    # Measure memory and CPU usage after adding delay
    memory_usage_after_delay = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_delay = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Delay Addition Time: {delay_time:.2f} seconds")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Adding Delay: {memory_usage_after_delay:.2f} MB (Change: {memory_usage_after_delay - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Adding Delay: {cpu_usage_after_delay:.2f}%")
