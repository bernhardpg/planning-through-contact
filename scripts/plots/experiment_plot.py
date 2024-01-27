import cv2
import matplotlib.pyplot as plt


# Function to convert a time in seconds to a frame index
def time_to_frame_index(time_sec, fps):
    return int(time_sec * fps)


video_path = "front_camera.mp4"  # Replace with your video path


def make_frames(video_path):
    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Ensure the video is opened
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Specify the times (in seconds) at which you want to extract frames
    times = [10, 35, 50, 70, 86.5, 95]  # Example times in seconds

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
        overlay_frame = cv2.addWeighted(frame1, alpha, frame2, 1, 0)

        frames.append(overlay_frame)

    # Clean up
    cap.release()

    return frames


frames_1 = make_frames("front_camera.mp4")

frames_per_row = len(frames_1)

# Create the figure with two rows of subplots
fig, axes = plt.subplots(
    1, frames_per_row, figsize=(20, 10)
)  # Adjust the size as needed

for col, frame in enumerate(frames_1):
    axes[col].imshow(frame)
    axes[col].axis("off")

# frames_1 = make_frames("front_camera.mp4")
# frames_2 = make_frames("front_camera.mp4")
#
# frames_per_row = len(frames_1)
#
# # Create the figure with two rows of subplots
# fig, axes = plt.subplots(
#     2, frames_per_row, figsize=(20, 10)
# )  # Adjust the size as needed
#
# for row, frames in enumerate([frames_1, frames_2]):
#     for col, frame in enumerate(frames):
#         axes[row, col].imshow(frame)
#         axes[row, col].axis("off")
#
#
plt.tight_layout()
plt.show()
