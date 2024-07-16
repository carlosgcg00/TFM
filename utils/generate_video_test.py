import cv2
import os
from glob import glob
import numpy as np

def zoom_effect(frame, zoom_factor):
    # Get dimensions of the frame
    height, width = frame.shape[:2]

    # Calculate the center of the frame
    center_x, center_y = width // 2, height // 2

    # Calculate new dimensions
    new_width, new_height = int(width / zoom_factor), int(height / zoom_factor)

    # Calculate cropping coordinates
    x1, y1 = center_x - new_width // 2, center_y - new_height // 2
    x2, y2 = center_x + new_width // 2, center_y + new_height // 2

    # Crop and resize the frame
    cropped_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(cropped_frame, (width, height))

    return zoomed_frame

def create_video_with_zoom(image_folder, output_video_path, frame_duration=2, fps=30, enable_zoom=False, zoom_in=True, zoom_speed=1.01, initial_zoom_factor=2.0):
    # Get all .jpg images in the folder
    image_paths = sorted(glob(os.path.join(image_folder, '*.jpg')))

    # Read the first image to get dimensions
    frame = cv2.imread(image_paths[0])
    height, width, layers = frame.shape

    # Video configuration
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the video file
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Number of frames per image
    frame_count = int(fps * frame_duration)

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        zoom_factor = initial_zoom_factor if zoom_in else 1.0

        for _ in range(frame_count):
            if enable_zoom:
                if zoom_in:
                    zoom_factor /= zoom_speed  # Apply zoom in effect
                else:
                    zoom_factor *= zoom_speed  # Apply zoom out effect
                    if zoom_factor > initial_zoom_factor:
                        zoom_factor = initial_zoom_factor

                zoomed_frame = zoom_effect(frame, zoom_factor)
                video.write(zoomed_frame)
            else:
                video.write(frame)

        # Toggle the zoom direction for the next image
        if enable_zoom:
            zoom_in = not zoom_in

    video.release()
    print(f"Video saved at: {output_video_path}")

if __name__ == "__main__":
    image_folder = 'D:\\Usuarios\\Carlos\\Documentos\\MUIT\\TFM\\Code\\AirbusAircraft\\archive\\test'
    output_video_path = 'test/test_video3_2_zoom_in_out.mp4'
    create_video_with_zoom(image_folder, output_video_path, enable_zoom=True, zoom_in=True, initial_zoom_factor=2.0)
