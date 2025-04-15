# SISTAR: Using ML to Detect DDoS Attacks and Pushback Defense in Programmable Data Plane
Meaning of SISTAR: switches are connected to each other like sisters in the network, and they light up the network like stars.


Note: This code is for experimental use only, don't use it directly in production

# Introduction


SISTAR is mainly used in programmable switches. By deploying machine learning on the programmable data plane and combining the characteristics of programmable switches, it realizes the protection of DDoS attacks.


The content provided by this repository is divided into 3 main sections:

1. model

2. BMv2

3. tofino

## 1. model

-----------------------------------------------
The following features are available:
|feature|reason|
|-------|--------|
|Destination Port|DDoS attacks are usually focused on a specific target port. Monitoring the traffic on the target port can help identify the attack. |
|Total Fwd Packets and Total Backward Packets|DDoS attacks usually result in an abnormal increase in the number of traffic packets. Monitoring the total number of forward and backward packets can help identify the attack. |
|Flow Duration|DDoS attacks may cause abnormally long or short flow durations. Monitoring the flow durations can help identify abnormal flows. |
|Flow Bytes/s and Flow Packets/s|DDoS attacks cause an abnormal increase in traffic rate. Monitoring the number of bytes and packets per second can help identify attacks. |
|Fwd Packet Length Max, Min, Mean, Std and Bwd Packet Length Max, Min, Mean, Std|DDoS attacks may cause abnormal packet length distribution. Monitoring the maximum, minimum, mean, and standard deviation of the forward and backward packet lengths can help identify attacks. |
|Min Packet Length and Max Packet Length|DDoS attacks may cause abnormal packet length distribution. Monitoring the minimum and maximum packet length can help identify attacks. |
|Average Packet Size|DDoS attacks may cause the average packet size to be abnormal. Monitoring the average packet size can help identify the attack. |
|Flow IAT Max, Min, Mean, Std| The mean and standard deviation of private arrival time (IAT) can reflect the sudden traffic. DDoS attacks may cause abnormal IAT distribution. Monitoring these characteristics can help identify attacks. |
|Fwd Packets/s and Bwd Packets/s| Abnormal increases in forward and backward packet rates can be characteristic of DDoS attacks, and monitoring these rates can help identify attacks. |
|Init_Win_bytes_forward and Init_Win_bytes_backward| Anomalies in the initial window size may be characteristics of some types of DDoS attacks. Monitoring these characteristics can help identify attacks. |
|act_data_pkt_fwd| The number of forward packets indicates traffic anomalies. DDoS attacks may increase the number of forward packets. Monitoring this feature can help identify attacks. |

Dataset:

|Dataset|Attack Types Covered|Coverage Scenarios|
|-------|--------|--------|
|CIC-IDS2017| DoS (GoldenEye, Hulk, Slowloris, Slowhttptest), DDoS (HTTP Flood, LOIC) |Simulated enterprise network environment, including short-term burst attacks and sustained suppression attacks.|
|CIC-IDS2018| DDoS (HOIC, LOIC), DoS (Slowloris, TCP Flood) | APT attack chains in complex enterprise networks, covering multi-stage mixed attack scenarios.|
|CIC-DDoS2019|   DDoS variants (SYN Flood, UDP Flood, ICMP Flood, HTTP Flood, Memcached reflection attacks) | High-intensity DDoS scenarios targeting cloud services, including Tb-level traffic attacks and coordinated attacks by IoT botnets.|
|CICIoT2023|  IoT botnet DDoS (Mirai variants, TCP/UDP flooding), Protocol vulnerability DoS (CoAP/Modbus) | Industrial IoT and smart home scenarios, covering attack topologies of 105 real IoT devices.|
|IoT23|  Botnet DDoS (Mirai, Gafgyt), DNS tunneling attacks | Smart home devices acting as attack sources, simulating distributed attacks launched after device hijacking.|
|UNSW-NB15|  Traditional DoS (SYN Flood, UDP Flood), Exploit-based DoS | University campus network environment, covering basic protocol layer attacks (TCP/UDP) and anomaly traffic detection. |

## 2. BMv2


## 3. tofino


# Code Architecture

```
-- model
    -- DT-CTS.py 

-- BMv2
    -- DT.p4 (implementation of P4 data plane, test the detection effect of DDoS attacks)
    -- topology.json (Experimental network topology connection)
    -- send.py (test send packet)
    -- receive.py (test receive packet)

-- tofino
    tna_detection.p4
```

## Environment
We use the software compiler to [p4c](https://github.com/p4lang/p4c), the simulation software switch [BMv2](https://github.com/p4lang/behavioral-model) to test, Through [p4runtime](https://github.com/p4lang/p4runtime) as our simple control plane.

You can use the following guide to get the complete environment installation [guide](https://github.com/jafingerhut/p4-guide)
