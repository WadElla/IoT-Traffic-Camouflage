import pyshark
import pandas as pd
import os

def process_pcap(file_path):
    # Load the pcap file
    cap = pyshark.FileCapture(file_path, keep_packets=True)

    # Dictionary to hold stream data
    stream_data = {}

    # Process each packet
    for packet in cap:
        try:
            # Check for TCP or UDP layer and process accordingly
            if 'TCP' in packet or 'UDP' in packet:
                protocol = 'TCP' if 'TCP' in packet else 'UDP'
                layer = packet['tcp'] if protocol == 'TCP' else packet['udp']
                stream_key = f"{protocol}_{layer.stream}"  # Create a compound key for the stream

                # Initialize the stream data structure if it doesn't exist
                if stream_key not in stream_data:
                    stream_data[stream_key] = {
                        'packets': [],
                        'total_length': 0,  # Initialize total length of conversation
                        'total_packet_length': 0,  # Initialize total packet length per stream
                        'times': []  # List to hold packet times for stream
                    }

                # Append packet to the list for this stream
                stream_data[stream_key]['packets'].append(packet)

                # Calculate segment length based on TCP or UDP
                if protocol == 'TCP':
                    segment_length = int(layer.len) if hasattr(layer, 'len') else 0
                else:
                    # Calculate UDP payload by subtracting the header length (8 bytes) from total UDP length
                    segment_length = max(0, int(layer.length) - 8) if hasattr(layer, 'length') else 0

                stream_data[stream_key]['total_length'] += segment_length
                packet_length = int(packet.length)
                stream_data[stream_key]['total_packet_length'] += packet_length

                # Append packet time to the list for this stream
                stream_data[stream_key]['times'].append(packet.sniff_time.timestamp())

        except AttributeError as e:
            print(f"Attribute error: {e} for packet {packet.number}")
        except Exception as e:
            print(f"General error: {e} for packet {packet.number}")

    # Prepare data for DataFrame
    rows = []
    for stream_key, data in stream_data.items():
        num_packets = len(data['packets'])
        times = data['times']
        if num_packets > 0:  # Ensure there are packets in the stream
            flow_time = times[-1] - times[0]  # Calculate flow time for the stream
            for i, packet in enumerate(data['packets']):
                if 'TCP' in packet:
                    segment_length = int(packet.tcp.len) if hasattr(packet.tcp, 'len') else 0
                else:
                    segment_length = max(0, int(packet.udp.length) - 8) if hasattr(packet.udp, 'length') else 0

                stream_time_diff = 0
                if i > 0:  # For packets after the first one, calculate stream time difference
                    stream_time_diff = times[i] - times[i - 1]

                rows.append({
                    'No.': int(packet.number),  # Packet number from pcap
                    'stream_id': stream_key,
                    'length': packet.length,
                    'total_packets': num_packets,
                    'conversation_length': data['total_length'],
                    'total_packet_length': data['total_packet_length'],
                    'TCP or UDP segment length': segment_length,  # Correct segment length for individual packet
                    'TCP or UDP stream time': stream_time_diff,  # Time difference between packets in the stream
                    'Flow time': flow_time  # Time difference between first and last packet in the stream
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Sort by packet number to retain original order
    df.sort_values('No.', inplace=True)

    # Save to CSV
    df.to_csv('stream_data.csv', index=False)

    # Close the capture file to release resources
    cap.close()



# usage
process_pcap('constant_size_delay_frag/st_repadxor/light_st_repadxor.pcap') ####################################

df1 = pd.read_csv('stream_data.csv')

df = pd.read_csv('constant_size_delay_frag/st_repadxor/light_st_repadxor.csv', encoding_errors='replace') #################################### from wireshark

# Convert 'Time' to seconds and perform calculations
df['Second'] = df['Time'].apply(lambda x: int(x))
#df['Second'] = df['Time'].astype(int)
packets_per_second = df.groupby('Second').size()
df['Packets per Second'] = df['Second'].map(packets_per_second)
df['Label'] = 4  ###################################################################

#print(df.columns)
df_final = df.drop(columns=['Time', 'Source', 'Destination', 'Info', 'Second', 'Protocol', 'TCP stream ','Length','TCP Segment Len'])

df2=df1.drop(columns=['No.', 'stream_id'])

result = pd.concat([df_final, df2], axis=1)

#df_final.dropna(inplace=True)
result = result.fillna(0)
#df_final['Packet count per flow'] = None
#df_final['Total segment length per flow'] = None
#print(df_final.head())
# Save the modified DataFrame to a new CSV file
result.to_csv('constant_size_delay_frag/dataset/light_st_repadxor.csv', index=False) ###################################

os.remove('stream_data.csv')
