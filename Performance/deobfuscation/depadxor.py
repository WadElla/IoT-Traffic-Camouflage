from scapy.all import wrpcap, rdpcap, IP, TCP, UDP, Raw
import time
import psutil
import os

def deobfuscate_payload(packet):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        obfuscated_payload = bytes(packet[layer].payload)

        # Extract the metadata from the end of the payload
        random_bytes_length = int.from_bytes(obfuscated_payload[-3:-1], 'big')
        pad_length = obfuscated_payload[-1]
        random_bytes = obfuscated_payload[-(3 + random_bytes_length):-3]
        xor_payload = obfuscated_payload[:-(3 + random_bytes_length)]

        # XOR the payload with the random bytes
        original_padded_payload = bytes(a ^ b for a, b in zip(xor_payload, random_bytes))

        # Remove the padding bytes
        original_payload = original_padded_payload[:-pad_length]

        packet[layer].remove_payload()
        packet.add_payload(Raw(original_payload))

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

def deobfuscate_traffic(input_pcap, output_pcap):
    packets = rdpcap(input_pcap)
    deobfuscated_packets = []
    total_padding_bytes_removed = 0

    start_time = time.time()  # Record start time

    for packet in packets:
        try:
            deobfuscated_packet, pad_length = deobfuscate_payload(packet)
            deobfuscated_packet.time = time.time()
            deobfuscated_packets.append(deobfuscated_packet)
            total_padding_bytes_removed += pad_length
        except Exception as e:
            print(f"Error removing padding and XOR from packet {packet.number}: {e}")

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, deobfuscated_packets)
    return total_padding_bytes_removed, latency

if __name__ == "__main__":
    input_pcap = 'performance/padxor.pcap'
    deobfuscated_pcap = 'performance/deobfuscated.pcap'

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform deobfuscation on the obfuscated packets
    print("Performing deobfuscation on obfuscated packets...")
    start_time = time.time()
    total_padding_bytes_removed, latency = deobfuscate_traffic(input_pcap, deobfuscated_pcap)
    end_time = time.time()
    deobfuscation_time = end_time - start_time
    print(f"Deobfuscated packets saved to: {deobfuscated_pcap}")

    # Measure memory and CPU usage after deobfuscation
    memory_usage_after_deobfuscation = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_deobfuscation = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Deobfuscation Time: {deobfuscation_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Padding Bytes Removed: {total_padding_bytes_removed} bytes")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Deobfuscation: {memory_usage_after_deobfuscation:.2f} MB (Change: {memory_usage_after_deobfuscation - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Deobfuscation: {cpu_usage_after_deobfuscation:.2f}%")
