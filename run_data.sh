#!/bin/bash
# spatial-temporal forecasting
python data_preparation/METR-LA/generate_training_data.py --history_seq_len 12 --future_seq_len 12
