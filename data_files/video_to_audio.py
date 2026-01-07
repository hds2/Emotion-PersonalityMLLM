from moviepy import VideoFileClip
import os

# Path to the folder containing video files
video_folder = r"D:\muti_data\Multi_data\data"

# Path to the output folder for extracted audio files
audio_output_folder = r"D:\muti_data\Multi_data\data_audios"
os.makedirs(audio_output_folder, exist_ok=True)  # Create the output directory if it does not exist

# Supported video file extensions
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# Iterate over all files in the video folder
for filename in os.listdir(video_folder):
    file_path = os.path.join(video_folder, filename)

    # Check if it is a file (not a directory)
    if not os.path.isfile(file_path):
        continue

    # Get the file extension (lowercased)
    ext = os.path.splitext(filename)[1].lower()

    # Skip non-video files
    if ext not in video_extensions:
        print(f"Skipping non-video file: {filename}")
        continue

    try:
        print(f"Processing video file: {filename}")

        # Load the video file
        video = VideoFileClip(file_path)

        # Build the output audio file path
        audio_filename = os.path.splitext(filename)[0] + ".mp3"
        audio_path = os.path.join(audio_output_folder, audio_filename)

        # Extract and save audio as MP3
        video.audio.write_audiofile(audio_path, codec="mp3")

        print(f"Audio saved to: {audio_path}")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

    finally:
        # Release resources
        try:
            video.close()
        except Exception:
            pass
