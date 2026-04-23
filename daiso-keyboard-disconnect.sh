#!/bin/bash

MAC=$(bluetoothctl devices | grep "TS-BTK25-D" | awk '{print $2}') && bluetoothctl disconnect $MAC
