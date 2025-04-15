sleep 2
echo "Start write s1 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.5.5/24 => 08:00:00:00:05:55 2
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9090
echo ""

sleep 2
echo "Start write s5 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.5.5/24 => 08:00:00:00:05:55 7
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9094
echo ""

sleep 2
echo "Start write s7 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.5.5/24 => 08:00:00:00:05:55 5
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9096
echo ""

sleep 2
echo "Start write s9 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.5.5/24 => 08:00:00:00:05:55 4
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9098
echo ""

sleep 2

echo "Start write s1 register16 threshold..."

echo "
register_write MyIngress.threshold_register16 0 84
register_write MyIngress.threshold_register16 1 84
register_write MyIngress.threshold_register16 2 84
register_write MyIngress.threshold_register16 3 46031
register_write MyIngress.threshold_register16 4 46031
register_write MyIngress.threshold_register16 5 46031
register_read MyIngress.threshold_register16
" | simple_switch_CLI --thrift-port 9094

sleep 2

echo "Start write s1 register32 threshold..."

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
echo "Start write s1 DDoS ternary rules..."

echo "
table_add MyIngress.DDoS_ternary MyIngress.drop 0x0&&&0x0c3 => 1

table_dump MyIngress.DDoS_ternary
" | simple_switch_CLI --thrift-port 9090
echo ""

sleep 2

echo "Start write s5 register16 threshold..."

echo "
register_write MyIngress.threshold_register16 0 84
register_write MyIngress.threshold_register16 1 84
register_write MyIngress.threshold_register16 2 84
register_write MyIngress.threshold_register16 3 46031
register_write MyIngress.threshold_register16 4 46031
register_write MyIngress.threshold_register16 5 46031
register_read MyIngress.threshold_register16
" | simple_switch_CLI --thrift-port 9094

sleep 2

echo "Start write s5 register32 threshold..."

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
" | simple_switch_CLI --thrift-port 9094

sleep 2
echo "Start write s5 DDoS ternary rules..."

echo "
table_add MyIngress.DDoS_ternary MyIngress.drop 0x0&&&0x0c3 => 1
table_add MyIngress.DDoS_ternary MyIngress.drop 0x40&&&0xf3 => 2

table_dump MyIngress.DDoS_ternary
" | simple_switch_CLI --thrift-port 9094
echo ""