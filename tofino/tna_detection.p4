/*******************************************************************************
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (c) 2021 Intel Corporation
 *  All Rights Reserved.
 *
 *  This software and the related documents are Intel copyrighted materials,
 *  and your use of them is governed by the express license under which they
 *  were provided to you ("License"). Unless the License provides otherwise,
 *  you may not use, modify, copy, publish, distribute, disclose or transmit
 *  this software or the related documents without Intel's prior written
 *  permission.
 *
 *  This software and the related documents are provided as is, with no express
 *  or implied warranties, other than those that are expressly stated in the
 *  License.
 ******************************************************************************/


#include <core.p4>
#if __TARGET_TOFINO__ == 3
#include <t3na.p4>
#elif __TARGET_TOFINO__ == 2
#include <t2na.p4>
#else
#include <tna.p4>
#endif

#include "headers.p4"
#include "util.p4"


control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {

    // Define the registers
    Register<bit<32>>(NUM_ENTRIES) fwd_header_length_register;
    Register<bit<32>>(NUM_ENTRIES) packet_length_mean_register;
    Register<bit<32>>(NUM_ENTRIES) packet_length_max_register;
    Register<bit<32>>(NUM_ENTRIES) packet_length_min_register;
    Register<bit<32>>(NUM_ENTRIES) packet_count_register;
    Register<bit<48>>(NUM_ENTRIES) timestamp_register;
    Register<bit<16>>(6) threshold_register16;
    Register<bit<32>>(9) threshold_register32;

    action drop() {
        mark_to_drop();
    }

    action update_fwd_header_length() {
        bit<32> current_length;
        fwd_header_length_register.read(current_length, ig_md.flow_index);
        fwd_header_length_register.write(ig_md.flow_index, current_length + ig_md.fwd_header_length);
    }

    action calculate_flow_index() {
        bit<32> temp_flow_index;
        bit<32> seed = NUM_ENTRIES;
        hash(temp_flow_index, HashAlgorithm.crc32, 0, 
            { hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.tcp.src_port, hdr.tcp.dst_port, hdr.ipv4.protocol }, seed);
        ig_md.flow_index = temp_flow_index;
    }

    table flow_table {
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            hdr.tcp.src_port: exact;
            hdr.tcp.dst_port: exact;
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

        packet_length_mean_register.read(mean_length, ig_md.flow_index);
        packet_length_max_register.read(max_length, ig_md.flow_index);
        packet_length_min_register.read(min_length, ig_md.flow_index);
        packet_count_register.read(packet_count, ig_md.flow_index);
        timestamp_register.read(timestamp, ig_md.flow_index);

        mean_length = ((mean_length << 1) + ig_md.current_packet_length) >> 1;
        max_length = (max_length > ig_md.current_packet_length) ? max_length : ig_md.current_packet_length;
        min_length = (min_length < ig_md.current_packet_length) ? min_length : ig_md.current_packet_length;

        bit<48> current_timestamp = ig_intr_md.global_tstamp;
        bit<48> time_diff = current_timestamp - timestamp;

        if (time_diff >= 1000000000) {
            ig_md.packets_per_second = packet_count;
            packet_count = 1;
            timestamp = current_timestamp;
        } else {
            packet_count = packet_count + 1;
        }

        packet_length_mean_register.write(ig_md.flow_index, mean_length);
        packet_length_max_register.write(ig_md.flow_index, max_length);
        packet_length_min_register.write(ig_md.flow_index, min_length);
        packet_count_register.write(ig_md.flow_index, packet_count);
        timestamp_register.write(ig_md.flow_index, timestamp);

        ig_md.packet_length_mean = mean_length;
        ig_md.packet_length_max = max_length;
        ig_md.packet_length_min = min_length;
        ig_md.flow_packet_count = packet_count;
    }

    action read_fwd_header_length() {
        bit<32> current_length;
        fwd_header_length_register.read(current_length, ig_md.flow_index);
        ig_md.fwd_header_length = current_length;
    }
    
    table egress_flow_table {
        key = {
            hdr.ipv4.src_addr: exact;
            hdr.ipv4.dst_addr: exact;
            hdr.tcp.src_port: exact;
            hdr.tcp.dst_port: exact;
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
        ig_md.bin_feature[1:0] = ig_md.Destination_Port_class;
        ig_md.bin_feature[3:2] = ig_md.Init_Win_bytes_forward_class;
        ig_md.bin_feature[5:4] = ig_md.Fwd_Header_Length_class;
        ig_md.bin_feature[7:6] = ig_md.Packet_Length_Mean_class;
        ig_md.bin_feature[9:8] = ig_md.Flow_Packets_persecond_class;
    }

    action bad_packet_forward(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
    }

    action good_packet_forward(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
    }

    table DDoS_ternary {
        key = {
            ig_md.bin_feature: ternary;
        }
        actions = {
            bad_packet_forward;
            good_packet_forward;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }
    
    action class1() {
        bit<16> threshold0;
        bit<16> threshold1;
        bit<16> threshold2;
        
        threshold_register16.read(threshold0, 0);
        threshold_register16.read(threshold1, 1);
        threshold_register16.read(threshold2, 2);

        if (ig_md.dst_port <= threshold0) {
            ig_md.Destination_Port_class = 0;
        } else if (ig_md.dst_port <= threshold1) {
            ig_md.Destination_Port_class = 1;
        } else if (ig_md.dst_port <= threshold2) {
            ig_md.Destination_Port_class = 2;
        } else {
            ig_md.Destination_Port_class = 3;
        }
    }
    
    action class2() {
        bit<16> threshold3;
        bit<16> threshold4;
        bit<16> threshold5;
        
        threshold_register16.read(threshold3, 3);
        threshold_register16.read(threshold4, 4);
        threshold_register16.read(threshold5, 5);

        if (ig_md.ctrl == 0x02) {
            if (ig_md.tcp_window <= threshold3) {
                ig_md.Init_Win_bytes_forward_class = 0;
            } else if (ig_md.tcp_window <= threshold4) {
                ig_md.Init_Win_bytes_forward_class = 1;
            } else if (ig_md.tcp_window <= threshold5) {
                ig_md.Init_Win_bytes_forward_class = 2;
            } else {
                ig_md.Init_Win_bytes_forward_class = 3;
            }
        }
    }

    action class3() {
        bit<32> threshold6;
        bit<32> threshold7;
        bit<32> threshold8;
        
        threshold_register32.read(threshold6, 0);
        threshold_register32.read(threshold7, 1);
        threshold_register32.read(threshold8, 2);

        if (ig_md.fwd_header_length <= threshold6) {
            ig_md.Fwd_Header_Length_class = 0;
        } else if (ig_md.fwd_header_length <= threshold7) {
            ig_md.Fwd_Header_Length_class = 1;
        } else if (ig_md.fwd_header_length <= threshold8) {
            ig_md.Fwd_Header_Length_class = 2;
        } else {
            ig_md.Fwd_Header_Length_class = 3;
        }
    }

    action class4() {
        bit<32> threshold9;
        bit<32> threshold10;
        bit<32> threshold11;
        
        threshold_register32.read(threshold9, 3);
        threshold_register32.read(threshold10, 4);
        threshold_register32.read(threshold11, 5);

        if (ig_md.packet_length_mean <= threshold9) {
            ig_md.Packet_Length_Mean_class = 0;
        } else if (ig_md.packet_length_mean <= threshold10) {
            ig_md.Packet_Length_Mean_class = 1;
        } else if (ig_md.packet_length_mean <= threshold11) {
            ig_md.Packet_Length_Mean_class = 2;
        } else {
            ig_md.Packet_Length_Mean_class = 3;
        }
    }

    action class5() {
        bit<32> threshold12;
        bit<32> threshold13;
        bit<32> threshold14;
        
        threshold_register32.read(threshold12, 6);
        threshold_register32.read(threshold13, 7);
        threshold_register32.read(threshold14, 8);

        if (ig_md.packets_per_second <= threshold12) {
            ig_md.Flow_Packets_persecond_class = 0;
        } else if (ig_md.packets_per_second <= threshold13) {
            ig_md.Flow_Packets_persecond_class = 1;
        } else if (ig_md.packets_per_second <= threshold14) {
            ig_md.Flow_Packets_persecond_class = 2;
        } else {
            ig_md.Flow_Packets_persecond_class = 3;
        } 
    }

    apply {
        class1();

        class2();

        // Calculating the head length
        if (hdr.ethernet.isValid() && hdr.ipv4.isValid() && hdr.tcp.isValid()) {
            ig_md.fwd_header_length = 14 + (bit<32>)hdr.ipv4.ihl * 4 + (bit<32>)hdr.tcp.dataOffset * 4;
            // The hash value of the stream is computed as the register index
            calculate_flow_index();
            // Find and update the header length of the stream
            flow_table.apply();
        }

        // Gets the current packet length
        ig_md.current_packet_length = ig_intr_md.packet_length;
        // Updating traffic characteristics
        update_flow_stats();

        class4();

        class5();

        // Finds and reads the header length of a stream
        egress_flow_table.apply();

        class3();

        parser_feature();

        DDoS_ternary.apply();



    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
