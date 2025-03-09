#!/bin/bash

# Run the first command
python biobank_classification.py --config xgb_config_ht.yaml

# Check if the first command was successful
if [ $? -eq 0 ]; then
    echo "First script completed successfully. Running second script..."
    python biobank_classification.py --config lr_config_ht.yaml
else
    echo "First script failed. Second script will not run."
    exit 1
fi
