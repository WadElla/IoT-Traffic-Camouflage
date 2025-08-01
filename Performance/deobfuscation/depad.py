from scapy.all import wrpcap, rdpcap, IP, TCP, UDP, Raw
import time
import psutil
import os

def remove_padding(packet):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)

        # Assuming that the last byte of the payload is the padding length
        pad_length = payload[-1]
        original_payload = payload[:-pad_length-1]

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

def remove_padding_from_traffic(input_pcap, output_pcap):
    packets = rdpcap(input_pcap)
    deobfuscated_packets = []
    total_padding_bytes_removed = 0

    start_time = time.time()  # Record start time

    for packet in packets:
        try:
            deobfuscated_packet, pad_length = remove_padding(packet)
            deobfuscated_packet.time = time.time()
            deobfuscated_packets.append(deobfuscated_packet)
            total_padding_bytes_removed += pad_length
        except Exception as e:
            print(f"Error removing padding from packet {packet.number}: {e}")

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, deobfuscated_packets)
    return total_padding_bytes_removed, latency

if __name__ == "__main__":
    input_pcap = 'performance/pad.pcap'
    deobfuscated_pcap = 'performance/deobfuscated.pcap'

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform deobfuscation on the padded packets
    print("Performing deobfuscation on padded packets...")
    start_time = time.time()
    total_padding_bytes_removed, latency = remove_padding_from_traffic(input_pcap, deobfuscated_pcap)
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
