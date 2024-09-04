#!/bin/bash


declare -a tickers=("AAPL" "GOOGL" "AMZN" "NVDA" "META" "TSM" "WMT" "V" "MA")


for ticker in "${tickers[@]}"
do
   python main.py $ticker
   # or do whatever with individual element of the array
done