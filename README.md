# SISTAR: Using ML to Detect DDoS Attacks and Pushback Defense in Programmable Data Plane
Meaning of SISTAR: switches are connected to each other like sisters in the network, and they light up the network like stars.


Note: This repository is only for anonymous review.

# Introduction


SISTAR is mainly used in programmable switches. By deploying machine learning on the programmable data plane and combining the characteristics of programmable switches, it realizes the protection of DDoS attacks.


The content provided by this repository is divided into 3 main sections:

1. model

2. BMv2

3. tofino

## 1. model

### Quick Start Guide

1. Training the DT-CTS model
    ```
    python3 DT-CTS.py
    ```

2. You should see the following output
    ```
    Test set accuracy: 0.8550

    Feature threshold usage record:
    Feature 6 used threshold count: 2
    Feature 8 used threshold count: 3
    Feature 2 used threshold count: 1
    ```



-----------------------------------------------
The following features are available:
|Feature|Reason|
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

-----------------------------------------------

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

### Quick Start Guide
To run the code

1. Use or Create a BMV2 VM or machine. For example you may download the VM from here provided by P4 developer day: https://drive.google.com/file/d/1uf9upiDVtHTT2ZoFb_Ekt4X-vIQCSNvh/view?usp=sharing

2. Check if the examples are compiling and working in tutorials/exercises folder.

3. Copy the BMv2 folder to tutorials/exercises/ folder

4. cd to BMv2 folder

5. Run the BMv2 software switch: 
    ```
    p4@p4:~/tutorials/exercises/BMv2$ make
    ```

6. Adding test flow tables:
    ```
    p4@p4:~/tutorials/exercises/BMv2$ ./entry-h1-h2.sh
    ```

7. Now you can test the connectivity between h1 and h2:
    ```
    mininet> h1 ping h2
    ```

### Environment
We use the software compiler to [p4c](https://github.com/p4lang/p4c), the simulation software switch [BMv2](https://github.com/p4lang/behavioral-model) to test, Through [p4runtime](https://github.com/p4lang/p4runtime) as our simple control plane.

You can use the following [guide](https://github.com/jafingerhut/p4-guide) to get the complete environment installation.

## 3. tofino

### Quick Start Guide
tna_detection.p4 under your p4 path of a Barefoot tofino switch, such as tna_range_match.p4 in its original path

1. cd to directory 
    ```
    cd ~/mydir/build
    ```

2. Run the cmake command to configure the build environment:
    ```
    cmake $SDE/p4studio/ \
    -DCMAKE_INSTALL_PREFIX=$SDE/install \
    -DCMAKE_MODULE_PATH=$SDE/cmake \
    -DP4_NAME=tna_detection \
    -DP4_PATH=~/mydir/tna_detection/tna_detection.p4
    ```


3. Compiling programs
    ```
    make
    ```

4. Installer
    ```
    make install
    ```

### Test

Open the terminal in three different Windows

1. Run the tofino model
    ```
    cd $SDE`
    ./run_tofino_model.sh --arch tofino -p tna_detection
    ```

2. Run the switch
    ```
    cd $SDE`
    ./run_switchd.sh --arch tofino -p tna_detection
    ```

3. Run the python test program
    ```
    cd $SDE
    ./run_p4_tests.sh --arch tofino -p tna_detection -t ~/mydir/tna_detection
    ```

4. Generate test programs
    ```
    cd ~/mydir
    p4testgen --target tofino --arch tna --std p4-16 --test-backend PTF --seed 1000 --max-tests 10 --out-dir tna_detection tna_detection/tna_detection.p4 --track-coverage STATEMENTS
    ```

5. More detailed test information
    ```
    p4testgen --target tofino --arch tna --std p4-16 --test-backend PTF --seed 1000 --max-tests 10 --out-dir tna_detection tna_detection/tna_detection.p4 --track-coverage STATEMENTS --print-coverage  --print-performance-report --path-selection GREEDY_STATEMENT_SEARCH --print-steps
    ```

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