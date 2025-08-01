from scapy.all import rdpcap, wrpcap, IP, TCP, UDP, Raw
import random
import time
import os

def fragment_payload(packet):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        
        original_length = len(payload)
        
        # If the payload is zero, return the original packet
        if original_length == 0:
            return [packet]
        
        # Calculate the minimum fragment size as one-third of the original length
        min_frag_size = max(1, original_length // 3)

        # If the payload is too small to fragment, return the original packet
        if original_length <= min_frag_size:
            return [packet]

        fragments = []

        while True:
            # Determine a random size for the first fragment
            frag_size1 = random.randint(min_frag_size, original_length - min_frag_size)
            frag_size2 = original_length - frag_size1

            # Ensure both fragments are within the valid size range and neither is equal to the original length
            if frag_size1 != original_length and frag_size2 != original_length:
                break

        # Create the first fragment
        fragment1_data = payload[:frag_size1]
        fragment2_data = payload[frag_size1:]

    
        """
        # Add random padding to the second fragment
        padding_length = random.randint(1, 256)
        padding = os.urandom(padding_length)
        fragment2_data += padding
        """
        # Create the first fragment packet
        frag_packet1 = packet.copy()
        frag_packet1[layer].remove_payload()
        frag_packet1.add_payload(Raw(fragment1_data))
        
        # Delete checksums and lengths so Scapy recalculates them
        if IP in frag_packet1:
            del frag_packet1[IP].len
            del frag_packet1[IP].chksum
        
        if TCP in frag_packet1:
            del frag_packet1[TCP].chksum
        if UDP in frag_packet1:
            del frag_packet1[UDP].chksum
        
        frag_packet1 = frag_packet1.__class__(bytes(frag_packet1))  # Rebuild the packet
        fragments.append(frag_packet1)

        # Create the second fragment packet
        frag_packet2 = packet.copy()
        frag_packet2[layer].remove_payload()
        frag_packet2.add_payload(Raw(fragment2_data))
        
        # Delete checksums and lengths so Scapy recalculates them
        if IP in frag_packet2:
            del frag_packet2[IP].len
            del frag_packet2[IP].chksum
        
        if TCP in frag_packet2:
            del frag_packet2[TCP].chksum
        if UDP in frag_packet2:
            del frag_packet2[UDP].chksum
        
        frag_packet2 = frag_packet2.__class__(bytes(frag_packet2))  # Rebuild the packet
        fragments.append(frag_packet2)

        return fragments
    else:
        return [packet]

def fragment_traffic(input_pcap, output_pcap, min_delay=0.01, max_delay=0.1):
    packets = rdpcap(input_pcap)
    fragmented_packets = []
    
    for packet in packets:
        fragments = fragment_payload(packet)
        for fragment in fragments:
            random_delay = random.uniform(min_delay, max_delay)
            time.sleep(random_delay)  # Introduce actual delay
            fragment.time = time.time()  # Set the timestamp to the current time after the delay
            fragmented_packets.append(fragment)
            
    wrpcap(output_pcap, fragmented_packets)

if __name__ == "__main__":
    input_pcap = 'fragmentation/welcome_frag_nodel_ori.pcap'  
    output_pcap = 'fragmentation/welcome_frag_del.pcap'
    min_delay = 0.01  # Minimum delay in seconds (10ms)
    max_delay = 0.1   # Maximum delay in seconds (100ms)
    fragment_traffic(input_pcap, output_pcap, min_delay, max_delay)
