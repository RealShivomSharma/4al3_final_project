#!/bin/bash

# Update package list and install prerequisites
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y software-properties-common

    # Add deadsnakes PPA for newer Python versions
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update

    # Install Python 3.11.9
    sudo apt install -y python3.11

    # Verify the installation
    python3.11 --version

    # Add Python 3.11 to PATH
    echo "export PATH=\"/usr/bin/python3.11:\$PATH\"" >> ~/.bashrc
    source ~/.bashrc
elif [[ "$OSTYPE" == "msys" ]]; then
    # Windows installation using Chocolatey
    choco install python --version=3.11.9

    # Verify the installation
    python --version

    # Add Python 3.11 to PATH
    setx PATH "%PATH%;C:\Python311"
fi
