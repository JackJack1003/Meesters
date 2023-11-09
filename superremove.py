import os

folder_path = './'  # Replace with the path to your folder

# Ensure the folder path exists
if os.path.exists(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate through the files and remove those ending with ".txt"
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"Removed: {file}")
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"Removed: {file}")
        if file.endswith(".pkl"):
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                print(f"Removed: {file}")

    print("Files with '.txt' extension removed successfully.")
else:
    print(f"The specified folder path '{folder_path}' does not exist.")
