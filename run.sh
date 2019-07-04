export PATH=/home/kovacs19/.conda/envs/panbot:$PATH
source activate panbot
python3.5 /home/kovacs19/PAN-bot-detection/Models/main_final.py --input_dir $1 --output_dir $2
