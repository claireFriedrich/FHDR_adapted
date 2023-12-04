import os
import shutil

data_path = "C:/Users/fricl/OneDrive/Documents/Suisse/EPFL/Cours/MA1/ML/all_datasets"

sub_folders = ['train', 'test']
sub_sub_folders = ['HDR', 'LDR']

test_hdr = os.path.join(data_path, "raw_ref_video_hdr")
test_ldr = os.path.join(data_path, "raw_video_png")
train_hdr = os.path.join(data_path, "tile_ref_video_hdr")
train_ldr = os.path.join(data_path, "tile_ref_video_png")

image_folders = [test_hdr, test_ldr, train_hdr, train_ldr]

# create right folders
if not os.path.exists("./dataset_final"):
    print("Making final dataset directory")
    os.makedirs("./dataset_final")

for sub_folder in sub_folders:
    if not os.path.exists(f"./dataset_final/{sub_folder}"):
        print(f"Making {sub_folder} directory")
        os.makedirs(f"./dataset_final/{sub_folder}")
    
    for sub_sub_folder in sub_sub_folders:
        if not os.path.exists(f"./dataset_final/{sub_folder}/{sub_sub_folder}"):
            print(f"Making {sub_folder}/{sub_sub_folder} directory")
            os.makedirs(f"./dataset_final/{sub_folder}/{sub_sub_folder}")


# moving the files to the proper directory
for image_folder in image_folders:
    print(f"Processing {image_folder}")
    filenames = [fn for fn in os.listdir(image_folder)]
    print(f"{len(filenames)} files")

    count_test_hdr = 0
    count_test_ldr = 0
    count_train_hdr = 0
    count_train_ldr = 0

    for filename in filenames:
        src_path = os.path.join(image_folder, filename)

        if filename.endswith('_video.png'):
            dst_path = "./dataset_final/test/LDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_test_ldr += 1
        elif filename.endswith('_ref.hdr'):
            dst_path = "./dataset_final/test/HDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_test_hdr += 1
        elif "_ref-" and ".hdr" in filename:
            dst_path = "./dataset_final/train/HDR"
            if not(os.path.isfile(dst_path)):
                shutil.copy(src_path, dst_path)
                count_train_ldr += 1
        elif "_video-"
        
    print(f"{count_test_hdr} test HDR images and {count_test_ldr} test LDR images")