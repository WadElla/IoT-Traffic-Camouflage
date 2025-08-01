import pyshark
from scapy.all import wrpcap, IP, TCP, UDP, Raw, rdpcap
import random
import time
import os

def organize_streams(file_path):
    cap = pyshark.FileCapture(file_path, keep_packets=True, use_json=True, include_raw=True)
    stream_data = {}

    for packet in cap:
        try:
            if 'TCP' in packet or 'UDP' in packet:
                protocol = 'tcp' if 'tcp' in packet else 'udp' if 'udp' in packet else None
                if protocol is None:
                    continue
                layer = packet['tcp'] if protocol == 'tcp' else packet['udp']
                stream_key = f"{protocol}_{layer.stream}"

                if stream_key not in stream_data:
                    stream_data[stream_key] = []

                stream_data[stream_key].append(packet)

        except AttributeError as e:
            print(f"Attribute error: {e} for packet {packet.number}")
        except Exception as e:
            print(f"General error: {e} for packet {packet.number}")

    cap.close()
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

def pad_and_shift_payload(packet, min_length, max_length):
    if TCP in packet or UDP in packet:
        layer = TCP if TCP in packet else UDP
        payload = bytes(packet[layer].payload)
        
        pad_length = random.randint(min_length, max_length)
        padding_bytes = os.urandom(pad_length)
        
        padded_payload = payload + padding_bytes if payload else padding_bytes
        
        shift_amount = random.randint(1, max(1, len(padded_payload) - 1))
        shifted_payload = padded_payload[-shift_amount:] + padded_payload[:-shift_amount]

        packet[layer].remove_payload()
        packet.add_payload(Raw(shifted_payload))

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

def pad_and_shift_traffic(input_pcap, output_pcap, min_length, max_length):
    packets = rdpcap(input_pcap)
    padded_packets = []

    for packet in packets:
        padded_packet = pad_and_shift_payload(packet, min_length, max_length)
        padded_packet.time = time.time()
        padded_packets.append(padded_packet)

    wrpcap(output_pcap, padded_packets)

if __name__ == "__main__":
    input_pcap = 'padshift/smartthings.pcapng'
    organized_pcap = 'padshift/smartthings_ori.pcap'
    padded_pcap = 'padshift/smartthings_repadshift.pcap'
    min_length = 1  # Minimum padding length
    max_length = 128  # Maximum padding length

    # Step 1: Capture and organize streams
    print("Organizing streams and saving to PCAP file...")
    stream_data = organize_streams(input_pcap)
    save_streams_to_pcap(stream_data, organized_pcap)
    print(f"Organized packets saved to: {organized_pcap}")

    # Step 2: Perform padding and shifting on the organized packets
    print("Performing padding and shifting on organized packets...")
    pad_and_shift_traffic(organized_pcap, padded_pcap, min_length, max_length)
    print(f"Padded and shifted packets saved to: {padded_pcap}")
