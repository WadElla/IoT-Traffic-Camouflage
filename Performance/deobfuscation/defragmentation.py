from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
import time
import psutil
import os

def reassemble_payload(packet1, packet2):
    if TCP in packet1 and TCP in packet2:
        layer = TCP
    elif UDP in packet1 and UDP in packet2:
        layer = UDP
    else:
        return packet1

    payload1 = bytes(packet1[layer].payload)
    payload2 = bytes(packet2[layer].payload)

    reassembled_payload = payload1 + payload2

    packet1[layer].remove_payload()
    packet1.add_payload(Raw(reassembled_payload))

    if TCP in packet1:
        del packet1[TCP].chksum
    if UDP in packet1:
        del packet1[UDP].chksum
    if IP in packet1:
        del packet1[IP].len
        del packet1[IP].chksum

    packet1 = packet1.__class__(bytes(packet1))
    return packet1

def deobfuscate_fragmented_traffic(input_pcap, output_pcap):
    packets = rdpcap(input_pcap)
    reassembled_packets = []
    total_fragments_removed = 0

    start_time = time.time()  # Record start time

    i = 0
    while i < len(packets):
        packet1 = packets[i]
        if i + 1 < len(packets):
            packet2 = packets[i + 1]
            if (TCP in packet1 and TCP in packet2 and packet1[TCP].sport == packet2[TCP].sport and packet1[TCP].dport == packet2[TCP].dport) or \
               (UDP in packet1 and UDP in packet2 and packet1[UDP].sport == packet2[UDP].sport and packet1[UDP].dport == packet2[UDP].dport):
                reassembled_packet = reassemble_payload(packet1, packet2)
                reassembled_packets.append(reassembled_packet)
                total_fragments_removed += 1
                i += 2
                continue
        reassembled_packets.append(packet1)
        i += 1

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, reassembled_packets)
    return total_fragments_removed, latency

if __name__ == "__main__":
    input_pcap = 'performance/frag.pcap'
    output_pcap = 'performance/reassembled.pcap'

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform deobfuscation on the fragmented packets
    print("Performing deobfuscation on fragmented packets...")
    start_time = time.time()
    total_fragments_removed, latency = deobfuscate_fragmented_traffic(input_pcap, output_pcap)
    end_time = time.time()
    deobfuscation_time = end_time - start_time
    print(f"Reassembled packets saved to: {output_pcap}")

    # Measure memory and CPU usage after deobfuscation
    memory_usage_after_deobfuscation = process.memory_info().rss / 1024 ** 2
    cpu_usage_after_deobfuscation = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Deobfuscation Time: {deobfuscation_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Fragments Removed: {total_fragments_removed}")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Deobfuscation: {memory_usage_after_deobfuscation:.2f} MB (Change: {memory_usage_after_deobfuscation - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Deobfuscation: {cpu_usage_after_deobfuscation:.2f}%")
