conda activate llava
start_idx=$1
end_idx=$2
python llava_response.py --start_idx $start_idx --end_idx $end_idx