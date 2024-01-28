#!/bin/bash

[ "$(ls -A /home/autoclues)" ] || cp -R /home/dump/. /home/autoclues
cd /home/autoclues
chmod 777 ./scripts/*

printf '\n\nOPTIMIZATION\n\n'
python experiment/optimization_launcher.py

printf '\n\nSUMMARIZATION\n\n'
python experiment/results_processors/optimization_results_summarizer.py

printf '\n\nDIVERSIFICATION\n\n'
python experiment/results_processors/diversificator.py --experiment exp2 --cadence 900 --max_time 900