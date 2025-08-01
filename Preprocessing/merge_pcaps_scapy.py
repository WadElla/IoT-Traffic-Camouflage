from scapy.all import rdpcap, wrpcap, PcapWriter
import os

def combine_pcaps(output_file, input_files):
    """
    Combine multiple PCAP files into a single PCAP file.
    
    :param output_file: Path to the output PCAP file.
    :param input_files: List of paths to the input PCAP files.
    """
    writer = PcapWriter(output_file, append=True, sync=True)
    
    for input_file in input_files:
        packets = rdpcap(input_file)
        for packet in packets:
            writer.write(packet)
    
    writer.close()
    print(f"Combined PCAP files into {output_file}")

if __name__ == "__main__":
    input_pcaps = [ "WeMoLink/Setup-C-1-STA.pcap",
                    "WeMoLink/Setup-A-9-STA.pcap", 
                    "WeMoLink/Setup-A-2-STA.pcap", 
                    "WeMoLink/Setup-A-4-STA.pcap", 
                    "WeMoLink/Setup-A-15-STA.pcap",         
                    ]
    output_pcap = "wemo/wemo.pcap"
    
    combine_pcaps(output_pcap, input_pcaps)
