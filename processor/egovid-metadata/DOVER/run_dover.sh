conda activate dover
cd ./DOVER
start_idx=$1
end_idx=$2
python egovid_dover.py --start_idx $start_idx --end_idx $end_idx