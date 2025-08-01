from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
import random
import time
import os

def pad_to_constant_size(packet, constant_packet_size):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        
        # Calculate the padding size required
        current_length = len(payload)
        if current_length >= constant_packet_size:
            return packet  # If payload is already larger than the constant size, return the original packet
        
        padding_length = constant_packet_size - current_length
        
        # Add padding to the payload
        padding = os.urandom(padding_length)
        padded_payload = payload + padding
        
        # Create a new packet with the padded payload
        padded_packet = packet.copy()
        padded_packet[layer].remove_payload()
        padded_packet.add_payload(Raw(padded_payload))
        
        # Delete checksums and lengths so Scapy recalculates them
        if IP in padded_packet:
            del padded_packet[IP].len
            del padded_packet[IP].chksum
        
        if TCP in padded_packet:
            del padded_packet[TCP].chksum
        if UDP in padded_packet:
            del padded_packet[UDP].chksum
        
        padded_packet = padded_packet.__class__(bytes(padded_packet))  # Rebuild the packet
        return padded_packet
    else:
        return packet

def process_traffic(input_pcap, output_pcap, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    processed_packets = []
    
    if not packets:
        raise ValueError("No packets found in the input pcap file.")
    
    # Set the initial CONSTANT_PACKET_SIZE based on the size of the first packet
    constant_packet_size = len(bytes(packets[0]))
    
    for packet in packets:
        packet_size = len(bytes(packet))
        if packet_size > constant_packet_size:
            constant_packet_size = packet_size  # Update CONSTANT_PACKET_SIZE if a larger packet is encountered
        
        padded_packet = pad_to_constant_size(packet, constant_packet_size)
        random_delay = random.uniform(min_delay, max_delay)
        time.sleep(random_delay)  # Introduce actual delay
        padded_packet.time = time.time()  # Set the timestamp to the current time after the delay
        processed_packets.append(padded_packet)
            
    wrpcap(output_pcap, processed_packets)

if __name__ == "__main__":
    input_pcap = 'performance/test.pcap'  
    output_pcap = 'performance/con.pcap'
    min_delay = 0.01  # Minimum delay in seconds (10ms)
    max_delay = 0.2   # Maximum delay in seconds (100ms)
    process_traffic(input_pcap, output_pcap, min_delay, max_delay)




