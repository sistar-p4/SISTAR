#!/bin/bash

echo "set s1 mtu 10240"

sudo ifconfig s1-eth1 mtu 10240
sudo ifconfig s1-eth2 mtu 10240
sudo ifconfig s1-eth3 mtu 10240

sleep 2

echo "Start write register16 threshold..."

echo "
register_write MyIngress.threshold_register16 0 84
register_write MyIngress.threshold_register16 1 84
register_write MyIngress.threshold_register16 2 84
register_write MyIngress.threshold_register16 3 46031
register_write MyIngress.threshold_register16 4 46031
register_write MyIngress.threshold_register16 5 46031
register_read MyIngress.threshold_register16
" | simple_switch_CLI --thrift-port 9090

sleep 2

echo "Start write register32 threshold..."

echo "
register_write MyIngress.threshold_register32 0 6
register_write MyIngress.threshold_register32 1 660
register_write MyIngress.threshold_register32 2 660
register_write MyIngress.threshold_register32 3 1
register_write MyIngress.threshold_register32 4 1
register_write MyIngress.threshold_register32 5 1
register_write MyIngress.threshold_register32 6 426
register_write MyIngress.threshold_register32 7 81312
register_write MyIngress.threshold_register32 8 81312
register_read MyIngress.threshold_register32
" | simple_switch_CLI --thrift-port 9090

sleep 2

echo "Start write ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.2.2/24 => 08:00:00:00:02:22 2
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.3.3/24 => 08:00:00:00:03:33 3
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9090
