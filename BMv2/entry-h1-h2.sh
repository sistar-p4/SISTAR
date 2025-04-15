sleep 2
echo "Start write s1 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.2.2/24 => 08:00:00:00:02:22 2
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9090
echo ""

sleep 2
echo "Start write s2 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 2
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.2.2/24 => 08:00:00:00:02:22 1
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9091
echo ""

sleep 2
echo "Start write s5 ternary rules..."

echo "
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.1.1/24 => 08:00:00:00:01:11 1
table_add MyIngress.ipv4_lpm MyIngress.ipv4_forward 10.0.2.2/24 => 08:00:00:00:02:22 2
table_dump MyIngress.ipv4_lpm
" | simple_switch_CLI --thrift-port 9094
echo ""