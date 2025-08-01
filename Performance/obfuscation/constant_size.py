from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
import random
import time
import os
import psutil

def pad_to_constant_size(packet, constant_packet_size):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        
        current_length = len(payload)
        if current_length >= constant_packet_size:
            return packet, 0  # If payload is already larger than the constant size, return the original packet
        
        padding_length = constant_packet_size - current_length
        padding = os.urandom(padding_length)
        padded_payload = payload + padding
        
        # Append padding length as metadata
        padded_payload += padding_length.to_bytes(2, 'big')
        
        padded_packet = packet.copy()
        padded_packet[layer].remove_payload()
        padded_packet.add_payload(Raw(padded_payload))
        
        if IP in padded_packet:
            del padded_packet[IP].len
            del padded_packet[IP].chksum
        
        if TCP in padded_packet:
            del padded_packet[TCP].chksum
        if UDP in padded_packet:
            del padded_packet[UDP].chksum
        
        padded_packet = padded_packet.__class__(bytes(padded_packet))
        return padded_packet, padding_length
    else:
        return packet, 0

def process_traffic(input_pcap, output_pcap, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    processed_packets = []
    total_padding_bytes = 0
    
    if not packets:
        raise ValueError("No packets found in the input pcap file.")
    
    # Set the initial CONSTANT_PACKET_SIZE based on the size of the first packet
    constant_packet_size = len(bytes(packets[0]))

    start_time = time.time()  # Record start time

    for packet in packets:
        packet_size = len(bytes(packet))
        if packet_size > constant_packet_size:
            constant_packet_size = packet_size  # Update CONSTANT_PACKET_SIZE if a larger packet is encountered
        
        padded_packet, padding_length = pad_to_constant_size(packet, constant_packet_size)
        total_padding_bytes += padding_length
        random_delay = random.uniform(min_delay, max_delay)
        time.sleep(random_delay)  # Introduce actual delay
        padded_packet.time = time.time()  # Set the timestamp to the current time after the delay
        processed_packets.append(padded_packet)
    
    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, processed_packets)
    return total_padding_bytes, latency, constant_packet_size

if __name__ == "__main__":
    input_pcap = 'st_dataset/dcam.pcap'  
    output_pcap = 'st_dataset/st_frag/dcam_st_frag.pcap'
    min_delay = 0.01  # Minimum delay in seconds (10ms)
    max_delay = 0.1   # Maximum delay in seconds (200ms)

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform padding to constant size on the traffic
    print("Performing padding to constant size on traffic...")
    start_time = time.time()
    total_padding_bytes, latency, constant_packet_size = process_traffic(input_pcap, output_pcap, min_delay, max_delay)
    end_time = time.time()
    padding_time = end_time - start_time
    print(f"Padded packets saved to: {output_pcap}")

    # Measure memory and CPU usage after padding
    memory_usage_after_padding = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_padding = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Padding Time: {padding_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Padding Bytes Added: {total_padding_bytes} bytes")
    print(f"Constant Packet Size: {constant_packet_size} bytes")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Padding: {memory_usage_after_padding:.2f} MB (Change: {memory_usage_after_padding - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Padding: {cpu_usage_after_padding:.2f}%")
