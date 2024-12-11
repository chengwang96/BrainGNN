# aal3
python 01-fetch_data.py --atlas aal3 --dataset_name adhd200 --dataset_dir data
python 02-process_data.py --atlas aal3 --dataset_name adhd200 --dataset_dir data --nclass 2 --score Gender
CUDA_VISIBLE_DEVICES=0 python 03-main.py --atlas aal3 --dataset_name adhd200 --dataset_dir data --indim 166 --nroi 166 --nclass 2

# cc200
python 01-fetch_data.py --atlas cc200 --dataset_name adhd200 --dataset_dir data
python 02-process_data.py --atlas cc200 --dataset_name adhd200 --dataset_dir data --nclass 2 --score Gender
CUDA_VISIBLE_DEVICES=0 python 03-main.py --atlas cc200 --dataset_name adhd200 --dataset_dir data --indim 166 --nroi 166 --nclass 2

# dk
python 01-fetch_data.py --atlas dk --dataset_name adhd200 --dataset_dir data
python 02-process_data.py --atlas dk --dataset_name adhd200 --dataset_dir data --nclass 2 --score Gender
CUDA_VISIBLE_DEVICES=0 python 03-main.py --atlas dk --dataset_name adhd200 --dataset_dir data --indim 166 --nroi 166 --nclass 2

# ho
python 01-fetch_data.py --atlas ho --dataset_name adhd200 --dataset_dir data
python 02-process_data.py --atlas ho --dataset_name adhd200 --dataset_dir data --nclass 2 --score Gender
CUDA_VISIBLE_DEVICES=0 python 03-main.py --atlas ho --dataset_name adhd200 --dataset_dir data --indim 166 --nroi 166 --nclass 2