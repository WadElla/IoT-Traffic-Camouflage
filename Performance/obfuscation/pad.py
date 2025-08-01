from scapy.all import wrpcap, IP, TCP, UDP, Raw, rdpcap
import random
import time
import os
import psutil

def pad_payload(packet, min_length, max_length):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        pad_length = random.randint(min_length, max_length)
        padding_bytes = os.urandom(pad_length)

        new_payload = payload + padding_bytes + bytes([pad_length])

        packet[layer].remove_payload()
        packet.add_payload(Raw(new_payload))

        if TCP in packet:
            del packet[TCP].chksum
        if UDP in packet:
            del packet[UDP].chksum
        if IP in packet:
            del packet[IP].len
            del packet[IP].chksum

        packet = packet.__class__(bytes(packet))
        return packet, pad_length
    else:
        return packet, 0

def pad_traffic(input_pcap, output_pcap, min_length, max_length, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    padded_packets = []
    total_padding_bytes = 0

    start_time = time.time()  # Record start time

    for packet in packets:
        try:
            padded_packet, pad_length = pad_payload(packet, min_length, max_length)
            #random_delay = random.uniform(min_delay, max_delay)
            #time.sleep(random_delay)  # Introduce actual delay
            padded_packet.time = time.time()
            padded_packets.append(padded_packet)
            total_padding_bytes += pad_length
        except Exception as e:
            print(f"Error padding packet {e}")

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, padded_packets)
    return total_padding_bytes, latency

if __name__ == "__main__":
    input_pcap = 'st_dataset/wemo.pcap'
    padded_pcap = 'st_dataset/st_pad/wemo_st_repad.pcap'
    
    min_length = 1
    max_length = 128

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform padding on the packets
    print("Performing padding on packets...")
    start_time = time.time()
    total_padding_bytes, latency = pad_traffic(input_pcap, padded_pcap, min_length, max_length)
    end_time = time.time()
    pad_time = end_time - start_time
    print(f"Padded packets saved to: {padded_pcap}")

    # Measure memory and CPU usage after padding
    memory_usage_after_pad = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_pad = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Padding Time: {pad_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Padding Bytes Added: {total_padding_bytes} bytes")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Padding: {memory_usage_after_pad:.2f} MB (Change: {memory_usage_after_pad - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Padding: {cpu_usage_after_pad:.2f}%")
