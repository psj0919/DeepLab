import os
import shutil

def flatten_camera_images(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    dir_list = os.listdir(src_root)
    for sub_folder in dir_list:
        camera_dir = os.path.join(src_root, sub_folder, 'sensor_raw_data', 'camera')

        if not os.path.exists(camera_dir):
            continue

        for subdir, _, files in os.walk(camera_dir):
            for file in files:
                # if file.endswith(('.jpg', 'png')):
                if file.endswith(('.json')):
                    src_path = os.path.join(subdir, file)

                    folder_prefix = os.path.basename(os.path.normpath(sub_folder))
                    new_file_name = f"{folder_prefix}_{file}"

                    dst_path = os.path.join(dst_root, new_file_name)
                    shutil.copy2(src_path, dst_path)


if __name__ == '__main__':
    src_path = '/storage/sjpark/vehicle_data/Night_data/096.상용_자율주행차_야간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL'
    dst_path = '/storage/sjpark/vehicle_data/Dataset3/test_json'

    flatten_camera_images(src_path, dst_path)
