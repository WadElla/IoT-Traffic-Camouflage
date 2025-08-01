from scapy.all import wrpcap, IP, TCP, UDP, Raw, rdpcap
import random
import time
import os
import psutil

def pad_and_xor_payload(packet, min_length, max_length):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        pad_length = random.randint(min_length, max_length)
        padding_bytes = os.urandom(pad_length)
        padded_payload = payload + padding_bytes

        random_bytes = os.urandom(len(padded_payload))
        xor_payload = bytes(a ^ b for a, b in zip(padded_payload, random_bytes))

        # Append metadata (random_bytes length, random_bytes, pad_length)
       # metadata = len(random_bytes).to_bytes(2, 'big') + random_bytes + pad_length.to_bytes(1, 'big')
        #xor_payload += metadata

        packet[layer].remove_payload()
        packet.add_payload(Raw(xor_payload))

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

def pad_and_xor_traffic(input_pcap, output_pcap, min_length, max_length, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    processed_packets = []
    total_padding_bytes = 0

    start_time = time.time()  # Record start time

    for packet in packets:
        try:
            processed_packet, pad_length = pad_and_xor_payload(packet, min_length, max_length)
            #random_delay = random.uniform(min_delay, max_delay)
            #time.sleep(random_delay)  # Introduce actual delay
            processed_packet.time = time.time()
            processed_packets.append(processed_packet)
            total_padding_bytes += pad_length
        except Exception as e:
            print(f"Error processing packet {packet.number}: {e}")

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, processed_packets)
    return total_padding_bytes, latency

if __name__ == "__main__":
    input_pcap = 'st_dataset/light.pcap'
    padded_pcap = 'st_dataset/st_padxor/light_st_repadxor.pcap'
    
    min_length = 1
    max_length = 128

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform padding and XOR on the packets
    print("Performing padding and XOR on packets...")
    start_time = time.time()
    total_padding_bytes, latency = pad_and_xor_traffic(input_pcap, padded_pcap, min_length, max_length)
    end_time = time.time()
    pad_xor_time = end_time - start_time
    print(f"Padded and XOR-ed packets saved to: {padded_pcap}")

    # Measure memory and CPU usage after padding and XOR
    memory_usage_after_pad_xor = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_pad_xor = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Padding and XOR Time: {pad_xor_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Padding Bytes Added: {total_padding_bytes} bytes")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Padding and XOR: {memory_usage_after_pad_xor:.2f} MB (Change: {memory_usage_after_pad_xor - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Padding and XOR: {cpu_usage_after_pad_xor:.2f}%")