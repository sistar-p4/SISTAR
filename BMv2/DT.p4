/* -*- P4_16 -*- */
#include <core.p4>
#include <v1model.p4>

const bit<16> TYPE_IPV4 = 0x0800;
const bit<8> PROTO_TCP = 6;
const bit<8> PROTO_UDP = 17;


typedef bit<9>  egressSpec_t;
typedef bit<48> macAddr_t;
typedef bit<32> ip4Addr_t;
typedef bit<16> port_t;

#define NUM_ENTRIES 8192


header ethernet_t {
    macAddr_t dstAddr;
    macAddr_t srcAddr;
    bit<16>   etherType;
}

header ipv4_t {
    bit<4>    version;
    bit<4>    ihl;
    bit<8>    diffserv;
    bit<16>   totalLen;
    bit<16>   identification;
    bit<3>    flags;
    bit<13>   fragOffset;
    bit<8>    ttl;
    bit<8>    protocol;
    bit<16>   hdrChecksum;
    ip4Addr_t srcAddr;
    ip4Addr_t dstAddr;
}

header tcp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<32> seqNo;
    bit<32> ackNo;
    bit<4>  dataOffset;
    bit<3>  res;
    bit<3>  ecn;
    bit<6>  ctrl;
    bit<16> window;
    bit<16> checksum;
    bit<16> urgentPtr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> length_;
    bit<16> checksum;
}

struct metadata {
    bit<32> bin_feature; // total binary feature

    bit<16> dstPort;
    bit<2>  Destination_Port_class;

    bit<6>  ctrl;
    bit<16> tcp_window;
    bit<2>  Init_Win_bytes_forward_class;

    bit<32> fwd_header_length;
    bit<32> flow_index;
    bit<2>  Fwd_Header_Length_class;

    bit<32> current_packet_length;
    bit<32> packet_length_mean;
    bit<32> packet_length_max;
    bit<32> packet_length_min;
    bit<32> flow_packet_count;
    bit<32> packets_per_second;
    bit<2>  Packet_Length_Mean_class;
    bit<2>  Packet_Length_Max_class;
    bit<2>  Packet_Length_Min_class;
    bit<2>  Flow_Packets_persecond_class;
}

struct headers {
    ethernet_t      ethernet;
    ipv4_t          ipv4;
    tcp_t           tcp;
    udp_t           udp;   
}

parser MyParser(packet_in packet,
                out headers hdr,
                inout metadata meta,
                inout standard_metadata_t standard_metadata) {

    state start {
        transition parser_ethernet;
    }

    state parser_ethernet {
        packet.extract(hdr.ethernet);
        transition select (hdr.ethernet.etherType) {
            TYPE_IPV4 : parser_ipv4;
            default : accept;
        }
    }

    state parser_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            PROTO_TCP : parser_tcp;
            default: accept;
        }
    }

    state parser_tcp {
        packet.extract(hdr.tcp);
        meta.dstPort = hdr.tcp.dstPort;
        meta.ctrl = hdr.tcp.ctrl;
        meta.tcp_window = hdr.tcp.window;
        transition accept;
    }
    
    state parse_udp {
        packet.extract(hdr.udp);
        meta.dstPort = hdr.udp.dstPort;
        transition accept;
    }
}

control MyVerifyChecksum(inout headers hdr, inout metadata meta) {
    apply {  }
}

control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t standard_metadata) {
    // Define the registers
    register<bit<32>>(NUM_ENTRIES) fwd_header_length_register;
    register<bit<32>>(NUM_ENTRIES) packet_length_mean_register;
    register<bit<32>>(NUM_ENTRIES) packet_length_max_register;
    register<bit<32>>(NUM_ENTRIES) packet_length_min_register;
    register<bit<32>>(NUM_ENTRIES) packet_count_register;
    register<bit<48>>(NUM_ENTRIES) timestamp_register;
    
    register<bit<16>>(6) threshold_register16; // 5*3
    register<bit<32>>(9) threshold_register32; 

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action update_fwd_header_length() {
        bit<32> current_length;
        fwd_header_length_register.read(current_length, meta.flow_index);
        fwd_header_length_register.write(meta.flow_index, current_length + meta.fwd_header_length);
    }

    action calculate_flow_index() {
        bit<32> temp_flow_index;
        bit<32> seed = 0;  // Specify the bit width explicitly
        bit<32> seed2 = NUM_ENTRIES;  // Specify the bit width explicitly
        hash(temp_flow_index, HashAlgorithm.crc32, seed, 
            { hdr.ipv4.srcAddr, hdr.ipv4.dstAddr, hdr.tcp.srcPort, hdr.tcp.dstPort, hdr.ipv4.protocol }, seed2);
        meta.flow_index = temp_flow_index;
    }

    table flow_table {
        key = {
            hdr.ipv4.srcAddr: exact;
            hdr.ipv4.dstAddr: exact;
            hdr.tcp.srcPort: exact;
            hdr.tcp.dstPort: exact;
            hdr.ipv4.protocol: exact;
        }
        actions = {
            update_fwd_header_length;
            NoAction;
        }
        size = NUM_ENTRIES;
        default_action = NoAction();
    }

    action update_flow_stats() {
        bit<32> mean_length;
        bit<32> max_length;
        bit<32> min_length;
        bit<32> packet_count;
        bit<48> timestamp;

        packet_length_mean_register.read(mean_length, meta.flow_index);
        packet_length_max_register.read(max_length, meta.flow_index);
        packet_length_min_register.read(min_length, meta.flow_index);
        packet_count_register.read(packet_count, meta.flow_index);
        timestamp_register.read(timestamp, meta.flow_index);

        // Update mean_length using shift operation
        mean_length = ((mean_length << 1) + meta.current_packet_length) >> 1;

        // update max_length and min_length
        max_length = (max_length > meta.current_packet_length) ? max_length : meta.current_packet_length;
        min_length = (min_length < meta.current_packet_length) ? min_length : meta.current_packet_length;

        // Calculate the current timestamp
        bit<48> current_timestamp = standard_metadata.ingress_global_timestamp;

        // Calculate the time difference
        bit<48> time_diff = current_timestamp - timestamp;

        // Reset the packet count and timestamp if the time difference is more than 1 second (1000 ms)
        if (time_diff >= 1000) {
            meta.packets_per_second = packet_count;
            packet_count = 1;
            timestamp = current_timestamp;
        } else {
            // Otherwise, update the packet count
            packet_count = packet_count + 1;
        }

        // Write back to register
        packet_length_mean_register.write(meta.flow_index, mean_length);
        packet_length_max_register.write(meta.flow_index, max_length);
        packet_length_min_register.write(meta.flow_index, min_length);
        packet_count_register.write(meta.flow_index, packet_count);
        timestamp_register.write(meta.flow_index, timestamp);

        meta.packet_length_mean = mean_length;
        meta.packet_length_max = max_length;
        meta.packet_length_min = min_length;
        meta.flow_packet_count = packet_count;
    }

    action read_fwd_header_length() {
        bit<32> current_length;
        fwd_header_length_register.read(current_length, meta.flow_index);
        meta.fwd_header_length = current_length;
    }
    
    table egress_flow_table {
        key = {
            hdr.ipv4.srcAddr: exact;
            hdr.ipv4.dstAddr: exact;
            hdr.tcp.srcPort: exact;
            hdr.tcp.dstPort: exact;
            hdr.ipv4.protocol: exact;
        }
        actions = {
            read_fwd_header_length;
            NoAction;
        }
        size = NUM_ENTRIES;
        default_action = NoAction();
    }

    action parser_feature() {
        meta.bin_feature[1:0] = meta.Destination_Port_class;
        meta.bin_feature[3:2] = meta.Init_Win_bytes_forward_class;
        meta.bin_feature[5:4] = meta.Fwd_Header_Length_class;
        meta.bin_feature[7:6] = meta.Packet_Length_Mean_class;
        meta.bin_feature[9:8] = meta.Flow_Packets_persecond_class;
    }

    action bad_packet_forward(egressSpec_t port) {
        standard_metadata.egress_spec = port;
    }

    action good_packet_forward(egressSpec_t port) {
        standard_metadata.egress_spec = port;
    }

    table DDoS_ternary {
        key = {
            meta.bin_feature: ternary;
        }
        actions = {
            drop;
            bad_packet_forward;
            good_packet_forward;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    action classify16(out bit<2> class, bit<16> value, bit<16> threshold1, bit<16> threshold2, bit<16> threshold3) {
        if (value <= threshold1) {
            class = 0;
        } else if (value <= threshold2) {
            class = 1;
        } else if (value <= threshold3) {
            class = 2;
        } else {
            class = 3;
        }
    }

    action classify32(out bit<2> class, bit<32> value, bit<32> threshold1, bit<32> threshold2, bit<32> threshold3) {
        if (value <= threshold1) {
            class = 0;
        } else if (value <= threshold2) {
            class = 1;
        } else if (value <= threshold3) {
            class = 2;
        } else {
            class = 3;
        }
    }

    action ipv4_forward(macAddr_t dstAddr, egressSpec_t port) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.srcAddr = hdr.ethernet.dstAddr;
        hdr.ethernet.dstAddr = dstAddr;
        hdr.ipv4.ttl = hdr.ipv4.ttl - 1;
    }

    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            ipv4_forward;
            drop;
            NoAction;
        }
        size = 1024;
        default_action = drop();
    }

    apply {
        bit<16> threshold0;
        bit<16> threshold1;
        bit<16> threshold2;
        bit<16> threshold3;
        bit<16> threshold4;
        bit<16> threshold5;
        bit<32> threshold6;
        bit<32> threshold7;
        bit<32> threshold8;
        bit<32> threshold9;
        bit<32> threshold10;
        bit<32> threshold11;
        bit<32> threshold12;
        bit<32> threshold13;
        bit<32> threshold14;

        threshold_register16.read(threshold0, 0);
        threshold_register16.read(threshold1, 1);
        threshold_register16.read(threshold2, 2);
        threshold_register16.read(threshold3, 3);
        threshold_register16.read(threshold4, 4);
        threshold_register16.read(threshold5, 5);

        threshold_register32.read(threshold6, 0);
        threshold_register32.read(threshold7, 1);
        threshold_register32.read(threshold8, 2);
        threshold_register32.read(threshold9, 3);
        threshold_register32.read(threshold10, 4);
        threshold_register32.read(threshold11, 5);
        threshold_register32.read(threshold12, 6);
        threshold_register32.read(threshold13, 7);
        threshold_register32.read(threshold14, 8);

        // 1. Extract features Destination Port
        classify16(meta.Destination_Port_class, meta.dstPort, threshold0, threshold1, threshold2);

        // 2. Extract the feature Init_Win_bytes_forward
        if (meta.ctrl == 0x02) {
            classify16(meta.Init_Win_bytes_forward_class, meta.tcp_window, threshold3, threshold4, threshold5);
        }

        // 3. Storage Fwd Header Length
        // Calculating the head length
        if (hdr.ethernet.isValid() && hdr.ipv4.isValid() && hdr.tcp.isValid()) {
            meta.fwd_header_length = 14 + (bit<32>)hdr.ipv4.ihl * 4 + (bit<32>)hdr.tcp.dataOffset * 4;
            // Compute stream hash as register index
            calculate_flow_index();
            // Find and update the header length of the stream
            flow_table.apply();
        }

        // 4. The Packet Length correlation value is stored and calculated
        // Gets the current packet length
        meta.current_packet_length = standard_metadata.packet_length;
        // Updating traffic characteristics
        update_flow_stats();
        classify32(meta.Packet_Length_Mean_class, meta.packet_length_mean, threshold9, threshold10, threshold11);

        // 5. Extracting features Flow Packets/s
        classify32(meta.Flow_Packets_persecond_class, meta.packets_per_second, threshold12, threshold13, threshold14);

        // 3. Extracting features Fwd Header Length
        // Finds and reads the header length of a stream
        egress_flow_table.apply();
        classify32(meta.Fwd_Header_Length_class, meta.fwd_header_length, threshold6, threshold7, threshold8);

        parser_feature();

        // DDoS Detection phase
        DDoS_ternary.apply();

        ipv4_lpm.apply();
    }
}

control MyEgress(inout headers hdr,
                 inout metadata meta,
                 inout standard_metadata_t standard_metadata) {
	apply {  }
}

control MyComputeChecksum(inout headers hdr, inout metadata meta) {
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.totalLen,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.fragOffset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr 
            },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16);
    }
}

control MyDeparser(packet_out packet, in headers hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.tcp);
    }
}

V1Switch(
MyParser(),
MyVerifyChecksum(),
MyIngress(),
MyEgress(),
MyComputeChecksum(),
MyDeparser()
) main;
