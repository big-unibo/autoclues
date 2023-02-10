#!/bin/bash

printf '\n\nOPTIMIZATION\n\n'
python experiment/optimization_launcher.py

printf '\n\nSUMMARIZATION\n\n'
python experiment/results_processors/optimization_results_summarizer.py

printf '\n\nDIVERSIFICATION\n\n'
python experiment/results_processors/diversificator.py --experiment exp1