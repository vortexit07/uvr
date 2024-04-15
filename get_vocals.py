# Removes the background instrumentals of songs and saves the file

import warnings
warnings.filterwarnings("ignore")
import os
import separate as uvr
from tinytag import TinyTag


def rename_files_with_prefix(directory, prefix, new_name):
    # Get list of files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file
    for filename in files:
        # Check if the file starts with the specified prefix
        if filename.startswith(prefix):
            
            # Construct the full paths for the old and new filenames
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed '{filename}' to '{new_name}'")  

def delete_files_with_prefix(directory, prefix):
    # Get list of files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file
    for filename in files:
        # Check if the file starts with the specified prefix
        if filename.startswith(prefix):
            # Construct the full path for the file
            file_path = os.path.join(directory, filename)
            
            # Delete the file
            os.remove(file_path)
            print(f"Deleted '{filename}'")

working_dir = "R:/Big Programming Projects/Python/SoundsLike/Data"
dir_artists = working_dir + '/Artists'
dir_songs = working_dir + '/Songs'

# Iterate over each directory and its subdirectories and files
for directory_path, _, file_names in os.walk(dir_songs):
    # Iterate over each file in the current directory
    for filename in file_names:
        # Construct the full path to the file
        file_path = os.path.join(directory_path, filename)
            
        # Separated tracks location
        separated = dir_artists
        
        # Artist name
        artist = TinyTag.get(file_path).artist
        
        # Separate vocals and instruments
        uvr.separate_audio(file_path, separated)
        
        # Delete intruments file
        delete_files_with_prefix(separated, 'instrument')
        
        # Rename vocals file to artist name
        rename_files_with_prefix(separated, 'vocal', artist + '.wav')