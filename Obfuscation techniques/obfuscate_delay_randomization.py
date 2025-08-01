from scapy.all import rdpcap, wrpcap
import random
import time

def add_delay_to_traffic(input_pcap, output_pcap, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    delayed_packets = []
    
    for packet in packets:
        #random_delay = random.uniform(min_delay, max_delay)
        #time.sleep(random_delay)  # Introduce actual delay
        packet.time = time.time()  # Set the timestamp to the current time after the delay
        delayed_packets.append(packet)
            
    wrpcap(output_pcap, delayed_packets)

if __name__ == "__main__":
    input_pcap = 'original_time/welcome.pcapng'  
    output_pcap = 'original_time/welcome_ori_time.pcap'
    min_delay = 0.01  # Minimum delay in seconds (10ms)
    max_delay = 0.2   # Maximum delay in seconds (100ms)
    add_delay_to_traffic(input_pcap, output_pcap, min_delay, max_delay)
