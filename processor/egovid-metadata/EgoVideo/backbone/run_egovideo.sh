conda activate egovideo

start_idx=$1
end_idx=$2

python egovideo_clean.py --start_idx $start_idx --end_idx $end_idx