import sys
import os

sys.path.append(".")
sys.path.append("..")


from options.test_options_videos_v2 import TestOptions

def run():
    test_opts = TestOptions().parse()
    angles_string = ""
    for angle_y in test_opts.novel_view_angles:
        angles_string += (str(angle_y) + " ")
    command = f"python scripts/inference_v2.py --exp_dir {test_opts.exp_dir} --checkpoint_path {test_opts.checkpoint_path} --data_path {test_opts.data_path} --novel_view_angles {angles_string}--test_batch_size {test_opts.test_batch_size} --test_workers {test_opts.test_workers}"
    os.system(command)
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_videos = os.path.join(test_opts.exp_dir, 'inference_videos')
    os.makedirs(out_path_videos, exist_ok=True)
    os.system(f'ffmpeg -framerate {test_opts.frame_rate} -i {out_path_results}/%d_original.jpg {out_path_videos}/original.mp4')
    os.system(f'ffmpeg -framerate {test_opts.frame_rate} -i {out_path_results}/%d_same-view.jpg {out_path_videos}/same-view.mp4')
    for angle_y in test_opts.novel_view_angles:
        os.system(f'ffmpeg -framerate {test_opts.frame_rate} -i {out_path_results}/%d_{angle_y}.jpg {out_path_videos}/{angle_y}.mp4')


if __name__ == '__main__':
    run()
