import os

input_folder = os.path.join("dataset/50")
file_list = os.listdir(input_folder)
total_files = len(file_list)
NUMBER_OF_FOLDERS = int(total_files / 5000)

target_path = os.path.join("dataset/steren")


print("Creating {} folders for data...".format(NUMBER_OF_FOLDERS))

for i in range(0, NUMBER_OF_FOLDERS):

    m = str(i + 1)

    create_path = os.path.join(target_path, m)

    # Create target Directory if don't exist
    if not os.path.exists(create_path):
        os.mkdir(create_path)
    else:    
        print("Skipping directory: ", create_path)

for i, file in enumerate(file_list):

    folder_number = str((i % NUMBER_OF_FOLDERS) + 1)

    file_path = os.path.join(input_folder, file)

    target_name = os.path.join(target_path, folder_number, file)

    os.rename(file_path, target_name)