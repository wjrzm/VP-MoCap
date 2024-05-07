import cv2
import numpy as np

class VideoWriter:
    """
    Write frames to a video.

    Call `write_frame` to write a single frame.
    Call `close` to release resource.

    """

    def __init__(self, path, width, height, fps, codec='avc1'):
        """
        Parameters
        ----------
        path : str
          Path to the video.
        width : int
          Frame width.
        height : int
          Frame height.
        fps : int
          Video frame rate.
        codec : str, optional
          Video codec, by default H264.
        """
        self.fps = fps
        self.width = width
        self.height = height
        self.video = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        self.frame_idx = 0

    def write_frame(self, frame):
        """
        Write one frame.

        Parameters
        ----------
        frame : np.ndarray
          Frame to write.
        """
        self.video.write(np.flip(frame, axis=-1).copy())
        self.frame_idx += 1

    def close(self):
        """
        Release resource.
        """
        self.video.release()