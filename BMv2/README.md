# BMv2

This folder contains scripts and configurations for setting up and managing a BMv2 (Behavioral Model version 2) software switch. BMv2 is a reference P4 software switch used for testing and development of P4 programs.

## Files

### `entry.sh`
This script is used to configure the BMv2 switch. It performs the following tasks:
1. Sets the MTU (Maximum Transmission Unit) for specific interfaces.
2. Writes values to `threshold_register16` and `threshold_register32` registers.
3. Configures ternary rules for IPv4 forwarding.

### Usage

To execute the script, ensure you have the necessary permissions and dependencies installed, such as `simple_switch_CLI`. Run the script using the following command:

```bash
./entry.sh
```

### Dependencies

- BMv2 software switch
- `simple_switch_CLI` for interacting with the switch
- Proper network interface configuration

### Notes

- Modify the script as needed to match your specific network setup.
- Ensure the thrift port (`9090` in the script) matches the configuration of your BMv2 instance.
