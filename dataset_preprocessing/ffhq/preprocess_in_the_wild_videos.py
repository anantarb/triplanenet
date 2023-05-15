import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--frame_rate', type=int, default=30)
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

video_name = args.video_path.split("/")[-1]
os.makedirs(args.out_dir, exist_ok=True)
os.system(f"cp {args.video_path} {args.out_dir}/")
os.chdir(f'{args.out_dir}')
command = f"ffmpeg -i {video_name} -r {args.frame_rate} frame%d.png"
print(command)
os.system(command)
os.chdir('..')

command = f"python preprocess_in_the_wild.py --indir {args.out_dir}" 
print(command)
os.system(command)


