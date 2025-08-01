from scapy.all import wrpcap, IP, TCP, UDP, Raw, rdpcap
import random
import time
import os
import psutil

def pad_and_shift_payload(packet, min_length, max_length):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        
        pad_length = random.randint(min_length, max_length)
        padding_bytes = os.urandom(pad_length)
        
        padded_payload = payload + padding_bytes if payload else padding_bytes
        
        shift_amount = random.randint(1, max(1, len(padded_payload) - 1))
        shift_amount_bytes = shift_amount.to_bytes(2, 'big')  # Convert shift_amount to 2-byte value

        shifted_payload = padded_payload[-shift_amount:] + padded_payload[:-shift_amount]

        # Append metadata (pad_length and shift_amount_bytes) at the end
        final_payload = shifted_payload + bytes([pad_length]) + shift_amount_bytes

        packet[layer].remove_payload()
        packet.add_payload(Raw(final_payload))

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

def pad_and_shift_traffic(input_pcap, output_pcap, min_length, max_length, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    padded_packets = []
    total_padding_bytes = 0

    start_time = time.time()  # Record start time

    for packet in packets:
        try:
            padded_packet, pad_length = pad_and_shift_payload(packet, min_length, max_length)
            #random_delay = random.uniform(min_delay, max_delay)
            #time.sleep(random_delay)  # Introduce actual delay
            padded_packet.time = time.time()
            padded_packets.append(padded_packet)
            total_padding_bytes += pad_length
        except Exception as e:
            print(f"Error padding and shifting packet: {e}")

    end_time = time.time()  # Record end time
    latency = end_time - start_time  # Calculate latency

    wrpcap(output_pcap, padded_packets)
    return total_padding_bytes, latency

if __name__ == "__main__":
    input_pcap = 'st_dataset/light.pcap'
    padded_pcap = 'st_dataset/st_padshift/light_st_repadshift.pcap'
    min_length = 1  # Minimum padding length
    max_length = 128  # Maximum padding length

    process = psutil.Process(os.getpid())

    # Measure memory usage before processing
    memory_usage_before = process.memory_info().rss / 1024 ** 2
    cpu_usage_before = process.cpu_percent(interval=None)

    # Step: Perform padding and shifting on the packets
    print("Performing padding and shifting on packets...")
    start_time = time.time()
    total_padding_bytes, latency = pad_and_shift_traffic(input_pcap, padded_pcap, min_length, max_length)
    end_time = time.time()
    pad_shift_time = end_time - start_time
    print(f"Padded and shifted packets saved to: {padded_pcap}")

    # Measure memory and CPU usage after padding and shifting
    memory_usage_after_pad_shift = process.memory_info().rss / 1024 ** 2

    time.sleep(20)

    cpu_usage_after_pad_shift = process.cpu_percent(interval=None)

    # Results
    print("\nMetrics:")
    print(f"Padding and Shifting Time: {pad_shift_time:.2f} seconds")
    print(f"Latency: {latency:.2f} seconds")
    print(f"Total Padding Bytes Added: {total_padding_bytes} bytes")
    print(f"Memory Usage Before: {memory_usage_before:.2f} MB")
    print(f"Memory Usage After Padding and Shifting: {memory_usage_after_pad_shift:.2f} MB (Change: {memory_usage_after_pad_shift - memory_usage_before:.2f} MB)")
    print(f"CPU Usage Before: {cpu_usage_before:.2f}%")
    print(f"CPU Usage After Padding and Shifting: {cpu_usage_after_pad_shift:.2f}%")
