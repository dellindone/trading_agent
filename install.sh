#!/bin/bash
# Two-step install to work around fyers-apiv3's hard pin on aiohttp==3.9.3
# (growwapi requires aiohttp>=3.11.18, which conflicts with fyers' exact pin)

set -e

# Step 1: Install fyers-apiv3 without its dependencies (skips the aiohttp==3.9.3 pin)
echo "Installing fyers-apiv3 --no-deps ..."
pip install "fyers-apiv3>=3.1.11" --no-deps

# Step 2: Install everything else (growwapi will pull aiohttp>=3.11.18)
echo "Installing remaining requirements ..."
pip install -r requirements.txt

echo "Done."
