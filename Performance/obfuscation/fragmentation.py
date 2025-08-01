from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
import random
import time
import os
import psutil

def calculate_packet_size(packet):
    return len(bytes(packet))

def fragment_payload(packet):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        
        original_length = len(payload)
        
        if original_length == 0:
            return [packet]
        
        min_frag_size = max(1, original_length // 3)

        if original_length <= min_frag_size:
            return [packet]

        fragments = []

        while True:
            frag_size1 = random.randint(min_frag_size, original_length - min_frag_size)
            frag_size2 = original_length - frag_size1

            if frag_size1 != original_length and frag_size2 != original_length:
                break

        fragment1_data = payload[:frag_size1]
        fragment2_data = payload[frag_size1:]

        frag_packet1 = packet.copy()
        frag_packet1[layer].remove_payload()
        frag_packet1.add_payload(Raw(fragment1_data))
        
        if IP in frag_packet1:
            del frag_packet1[IP].len
            del frag_packet1[IP].chksum
        
        if TCP in frag_packet1:
            del frag_packet1[TCP].chksum
        if UDP in frag_packet1:
            del frag_packet1[UDP].chksum
        
        frag_packet1 = frag_packet1.__class__(bytes(frag_packet1))
        fragments.append(frag_packet1)

        frag_packet2 = packet.copy()
        frag_packet2[layer].remove_payload()
        frag_packet2.add_payload(Raw(fragment2_data))
        
        if IP in frag_packet2:
            del frag_packet2[IP].len
            del frag_packet2[IP].chksum
        
        if TCP in frag_packet2:
            del frag_packet2[TCP].chksum
        if UDP in frag_packet2:
            del frag_packet2[UDP].chksum
        
        frag_packet2 = frag_packet2.__class__(bytes(frag_packet2))
        fragments.append(frag_packet2)

        return fragments
    else:
        return [packet]

def fragment_traffic(input_pcap, output_pcap, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    fragmented_packets = []
    total_additional_bytes = 0

    start_time = time.time()  # Record start time

    for packet in packets:
        original_size = calculate_packet_size(packet)
        fragments = fragment_payload(packet)
        fragments_size = sum(calculate_packet_size(frag) for frag in fragments)
        additional_bytes = fragments_size - original_size
        total_additional_bytes += additional_bytes

        for fragment in fragments:
            #random_delay = random.uniform(min_delay, max_delay)
            #time.sleep(random_delay)
            fragment.time = time.time()
            fragmented_packets.append(fragment)

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, fragmented_packets)
    return total_additional_bytes, latency

if __name__ == "__main__":
    input_pcap = 'st_dataset/light.pcap'  
    output_pcap = 'st_dataset/st_frag/light_st_frag.pcap'
    min_delay = 0.01  # Minimum delay in seconds (10ms)
    max_delay = 0.1   # Maximum delay in seconds (100ms)

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform fragmentation on the traffic
    print("Performing fragmentation on traffic...")
    start_time = time.time()
    total_additional_bytes, latency = fragment_traffic(input_pcap, output_pcap, min_delay, max_delay)
    end_time = time.time()
    fragmentation_time = end_time - start_time
    print(f"Fragmented packets saved to: {output_pcap}")

    # Measure memory and CPU usage after fragmentation
    memory_usage_after_fragmentation = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_fragmentation = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Fragmentation Time: {fragmentation_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Additional Bytes Incurred: {total_additional_bytes} bytes")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Fragmentation: {memory_usage_after_fragmentation:.2f} MB (Change: {memory_usage_after_fragmentation - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Fragmentation: {cpu_usage_after_fragmentation:.2f}%")