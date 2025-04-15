sleep 2
echo "Start write s1 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.4.4/24 => 08:00:00:00:04:44 3
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9090
echo ""

sleep 2
echo "Start write s6 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.4.4/24 => 08:00:00:00:04:44 4
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9095
echo ""

sleep 2
echo "Start write s4 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 3
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.4.4/24 => 08:00:00:00:04:44 1
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9093
echo ""