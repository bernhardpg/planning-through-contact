import cv2
import matplotlib.pyplot as plt


# Function to convert a time in seconds to a frame index
def time_to_frame_index(time_sec, fps):
    return int(time_sec * fps)


video_paths = [
    # "traj_0_video.mp4",  # the one from shao slack
    "traj_1_video.mp4",  # the one from shao slack
    # "traj_14_video.mp4",  # the one from shao slack
]
main_folder = "videos/"

# Specify the times (in seconds) at which you want to extract frames
times_per_video = [
    # [20, 55, 72, 95, 111, 146, 162, 196, 209, 225],
    [20, 56, 73, 109, 125, 160],
    # [10, 35, 50, 70, 86.5, 95],
]


def make_frames(video_path, times):
    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Ensure the video is opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Extract and overlay frames
    frames = []
    for idx in range(len(times) - 1):
        time_sec = times[idx]
        frame_index = time_to_frame_index(time_sec, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame1 = cap.read()
        if not ret:
            break

        time_sec_next = times[idx + 1]
        frame_index = time_to_frame_index(time_sec_next, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame2 = cap.read()
        if not ret:
            frame2 = frame1  # If there's no next frame, use the same frame

        # Convert color to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        # Overlay the second frame on the first frame with transparency
        alpha = 0.5  # Transparency factor
        overlay_frame = cv2.addWeighted(frame1, alpha, frame2, 0.8, 0)

        frames.append(overlay_frame)

    # Clean up
    cap.release()

    return frames


frames_per_video = [
    # make_frames(main_folder + path, times[:-1])  # skip last frame
    make_frames(main_folder + path, times)  # skip last frame
    for path, times in zip(video_paths, times_per_video)
]
frames_per_row = max([len(frame) for frame in frames_per_video])
num_seqs = len(frames_per_video)

individual_plots = True
if individual_plots:
    for frames in frames_per_video:
        frames_per_row = len(frames)
        fig, axes = plt.subplots(
            1, frames_per_row, figsize=(1.9 * frames_per_row, 1.7)
        )  # Adjust the size as needed

        for col, frame in enumerate(frames):
            axes[col].imshow(frame)
            axes[col].axis("off")

else:
    for row, frames in enumerate(frames_per_video):
        for col, frame in enumerate(frames):
            axes[row, col].imshow(frame)

        # Make sure to turn of axis for ALL subplots, including
        # where there is no video frame
        for col in range(frames_per_row):
            axes[row, col].axis("off")

# Adjust subplot parameters to remove padding
plt.subplots_adjust(wspace=0, hspace=1.2)
plt.tight_layout()
plt.show()
