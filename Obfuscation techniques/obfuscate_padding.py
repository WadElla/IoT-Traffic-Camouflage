import pyshark
from scapy.all import wrpcap, IP, TCP, UDP, Raw, rdpcap
import random
import time
import os

def capture_streams(file_path):
    cap = pyshark.FileCapture(file_path, keep_packets=True, use_json=True, include_raw=True)
    stream_data = {}
    skipped_packets = []

    for packet in cap:
        try:
            if 'tcp' in packet or 'udp' in packet:
                protocol = 'tcp' if 'tcp' in packet else 'udp'
                layer = packet['tcp'] if protocol == 'tcp' else packet['udp']
                stream_key = f"{protocol}_{layer.stream}"

                if stream_key not in stream_data:
                    stream_data[stream_key] = []

                stream_data[stream_key].append(packet)
            else:
                print(f"Packet {packet.number} is not TCP or UDP.")
                skipped_packets.append(packet)

        except AttributeError as e:
            print(f"Attribute error: {e} for packet {packet.number}")
            skipped_packets.append(packet)
        except Exception as e:
            print(f"General error: {e} for packet {packet.number} - {e}")
            skipped_packets.append(packet)

    cap.close()
    print(f"Total packets skipped: {len(skipped_packets)}")
    return stream_data

def save_streams_to_pcap(stream_data, output_pcap):
    organized_packets = []
    for packets in stream_data.values():
        for packet in packets:
            try:
                raw_packet = packet.get_raw_packet()
                if raw_packet:
                    organized_packets.append(bytes(raw_packet))
            except Exception as e:
                print(f"Error processing packet {packet.number}: {e}")
    wrpcap(output_pcap, organized_packets)

def pad_payload(packet, min_length, max_length):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        pad_length = random.randint(min_length, max_length)
        padding_bytes = os.urandom(pad_length)

        if payload:
            new_payload = payload + padding_bytes
            packet[layer].remove_payload()
            packet.add_payload(Raw(new_payload))
        else:
            packet.add_payload(Raw(padding_bytes))

        if TCP in packet:
            del packet[TCP].chksum
        if UDP in packet:
            del packet[UDP].chksum
        if IP in packet:
            del packet[IP].len
            del packet[IP].chksum

        packet = packet.__class__(bytes(packet))
        return packet
    else:
        return packet

def pad_traffic(input_pcap, output_pcap, min_length, max_length):
    packets = rdpcap(input_pcap)
    padded_packets = []

    for packet in packets:
        try:
            padded_packet = pad_payload(packet, min_length, max_length)
            padded_packet.time = time.time()
            padded_packets.append(padded_packet)
        except Exception as e:
            print(f"Error padding packet {packet.number}: {e}")

    wrpcap(output_pcap, padded_packets)

if __name__ == "__main__":
    input_pcap = 'padding/galaxytab.pcap'
    organized_pcap = 'padding/galaxytab_ori.pcap'
    padded_pcap = 'padding/galaxytab_pad.pcap'
    
    min_length = 1
    max_length = 256

    # Step 1: Capture and organize streams
    stream_data = capture_streams(input_pcap)
    
    # Step 2: Save organized streams to a new PCAP file
    print("Saving organized streams to a new PCAP file...")
    save_streams_to_pcap(stream_data, organized_pcap)
    print(f"Organized packets saved to: {organized_pcap}")

    # Step 3: Perform padding on the organized packets
    print("Performing padding on organized packets...")
    pad_traffic(organized_pcap, padded_pcap, min_length, max_length)
    print(f"Padded packets saved to: {padded_pcap}")
