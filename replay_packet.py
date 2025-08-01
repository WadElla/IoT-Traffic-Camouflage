import pyshark
from scapy.all import sendp, sniff, Ether, rdpcap, wrpcap, IP, IPv6, TCP, UDP
import time
from threading import Thread, Event

def send_packet(packet, iface):
    raw_packet = bytes(packet)
    ether_packet = Ether(raw_packet)
    print(f"Sending packet: {ether_packet.summary()}")
    sendp(ether_packet, iface=iface, verbose=False)

def packet_callback(packet, captured_packets, expected_flows):
    if TCP in packet or UDP in packet:
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet[TCP].sport if TCP in packet else packet[UDP].sport
            dst_port = packet[TCP].dport if TCP in packet else packet[UDP].dport
        elif IPv6 in packet:
            src_ip = packet[IPv6].src
            dst_ip = packet[IPv6].dst
            src_port = packet[TCP].sport if TCP in packet else packet[UDP].sport
            dst_port = packet[TCP].dport if TCP in packet else packet[UDP].dport
        else:
            return

        if (src_ip, dst_ip, src_port, dst_port) in expected_flows or (dst_ip, src_ip, dst_port, src_port) in expected_flows:
            packet.time = time.time()
            print(f"Captured packet: {packet.summary()} with new timestamp: {packet.time}")
            captured_packets.append(packet)

def global_sniffer(iface, captured_packets, stop_sniffing, expected_flows):
    sniff(iface=iface, store=0, prn=lambda p: packet_callback(p, captured_packets, expected_flows), stop_filter=lambda p: stop_sniffing.is_set())

def replay_streams(file_path, iface):
    packets = rdpcap(file_path)
    captured_packets = []
    stop_sniffing = Event()
    expected_flows = set()

    # Collect all unique source and destination IP addresses and ports from the pcap file
    for packet in packets:
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            else:
                continue
        elif IPv6 in packet:
            src_ip = packet[IPv6].src
            dst_ip = packet[IPv6].dst
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            else:
                continue
        else:
            continue

        expected_flows.add((src_ip, dst_ip, src_port, dst_port))

    sniffer_thread = Thread(target=global_sniffer, args=(iface, captured_packets, stop_sniffing, expected_flows))
    sniffer_thread.start()

    # Allow time for the sniffer to start
    time.sleep(1)

    for packet in packets:
        send_packet(packet, iface)
        time.sleep(0.037)  # Slightly increased delay to help the sniffer keep up

    # Allow some time for any last packets to be captured
    time.sleep(5)
    stop_sniffing.set()
    sniffer_thread.join()

    return captured_packets

if __name__ == "__main__":
    input_pcap = 'printer.pcap'
    replay_pcap = 'replayed_packets.pcap'

    # Identify your network interface name (e.g., en0 on macOS)
    iface = 'en0'

    # Step 1: Replay streams and capture packets
    print("Replaying packets and capturing them...")
    captured_packets = replay_streams(input_pcap, iface)
    print(f"Captured {len(captured_packets)} packets.")

    # Debugging: Print details of captured packets
    for i, packet in enumerate(captured_packets):
        if not packet.haslayer(Ether):
            print(f"Packet {i} does not have an Ethernet layer.")
        else:
            print(f"Packet {i}: {packet.summary()}")

    # Save the captured packets to a new pcap file
    wrpcap(replay_pcap, captured_packets)

    print(f"Replayed packets saved to: {replay_pcap}")
