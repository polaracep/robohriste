import cv2
import numpy as np
import argparse
from collections import deque

# Default configuration values - easy to modify
DEFAULT_CAMERA_INDEX = 0  # Keep camera 1 as requested

# i corrected the default HSV values, do not change it

DEFAULT_HUE_MIN = 90
DEFAULT_HUE_MAX = 135

DEFAULT_SAT_MIN = 100
DEFAULT_SAT_MAX = 255

DEFAULT_VAL_MIN = 100
DEFAULT_VAL_MAX = 255


class PingPongBallTracker:
    def __init__(self, ball_color_lower, ball_color_upper, dividing_line_position=0.5):
        """
        Initialize the ping pong ball tracker.

        Args:
            ball_color_lower: Lower HSV threshold for the ball color detection
            ball_color_upper: Upper HSV threshold for the ball color detection
            dividing_line_position: Position of the dividing line (0.0-1.0, default 0.5 for middle)
        """
        self.ball_color_lower = np.array(ball_color_lower)
        self.ball_color_upper = np.array(ball_color_upper)
        self.dividing_line_position = dividing_line_position

        # Ball size consistency based on real setup: 106cm height, 39mm ball
        self.expected_ball_radius = 20  # Expected radius in pixels (can be calibrated)

        # Improved tracking parameters
        self.tracked_balls = []
        self.max_distance_for_tracking = 80
        self.tracking_history = 20  # Frames to keep track of disappeared balls
        self.track_paths = {}  # Store movement paths for better visualization
        self.path_length = 15  # Maximum path length to remember
        self.min_detection_confidence = 2  # Minimum detections before considering it a valid ball
        self.next_id = 0  # For assigning unique IDs

        # Ball crossing counter system
        self.total_balls_left = 10  # Initial balls on left side (adjustable)
        self.total_balls_right = 10  # Initial balls on right side (adjustable)
        self.crossing_detection_zone = 50
        self.ball_crossing_history = {}  # Track which side each ball was on
        self.crossing_cooldown = {}  # Prevent multiple crossings for same ball
        self.cooldown_frames = 15

    def update_hsv_range(self, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
        """Update HSV color range for ball detection."""
        self.ball_color_lower = np.array([hue_min, sat_min, val_min])
        self.ball_color_upper = np.array([hue_max, sat_max, val_max])

    def detect_balls(self, frame):
        """Detect ping pong balls using simple watershed algorithm."""
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for ball color
        mask = cv2.inRange(hsv, self.ball_color_lower, self.ball_color_upper)

        # Apply simple watershed-based ball detection
        detected_balls, processed_mask = self._simple_watershed_detection(frame, mask)

        return detected_balls, processed_mask

    def _simple_watershed_detection(self, original_frame, mask):
        """Simple and fast watershed algorithm for ping pong ball separation."""
        # Check if mask has any content
        if cv2.countNonZero(mask) == 0:
            return [], mask

        # Simple noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Distance transform
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

        # Find sure foreground - simple threshold
        _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

        # Find sure background
        sure_bg = cv2.dilate(closing, kernel, iterations=3)

        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg.astype(np.uint8))

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(original_frame, markers)

        # Extract balls from markers
        detected_balls = []
        for label in np.unique(markers):
            if label <= 1:  # Skip background
                continue

            # Create mask for this label
            label_mask = (markers == label).astype(np.uint8) * 255

            # Find contour
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)

            # Simple area filter
            if 50 < area < 2000:
                # Get ball center and radius
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center_x, center_y, radius = int(x), int(y), int(radius)

                # Simple radius filter
                if 5 < radius < 50:
                    # Simple circularity check
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Basic circularity filter
                            detected_balls.append((center_x, center_y, radius, circularity))

        return detected_balls, closing

    def update_tracking(self, detected_balls):
        """Simple ball tracking system."""
        # Initialize tracking if first time
        if not self.tracked_balls:
            for x, y, r, circ in detected_balls:
                self._add_new_ball(x, y, r, circ)
            return

        # Simple distance-based matching
        matched_detected = [False] * len(detected_balls)
        matched_tracked = [False] * len(self.tracked_balls)

        # Create distance assignments
        assignments = []
        for i, tracked_ball in enumerate(self.tracked_balls):
            if not tracked_ball["active"]:
                continue

            for j, (x, y, r, circ) in enumerate(detected_balls):
                distance = np.sqrt((tracked_ball["center"][0] - x) ** 2 + (tracked_ball["center"][1] - y) ** 2)
                assignments.append((distance, i, j))

        # Sort by distance and assign
        assignments.sort()

        for dist, tracked_idx, detected_idx in assignments:
            if (dist < self.max_distance_for_tracking and
                    not matched_tracked[tracked_idx] and
                    not matched_detected[detected_idx]):
                self._update_tracked_ball(tracked_idx, detected_balls[detected_idx])
                matched_detected[detected_idx] = True
                matched_tracked[tracked_idx] = True

        # Handle unmatched tracked balls
        for i, tracked_ball in enumerate(self.tracked_balls):
            if tracked_ball["active"] and not matched_tracked[i]:
                tracked_ball["ttl"] -= 1
                if tracked_ball["ttl"] <= 0:
                    tracked_ball["active"] = False

        # Add new balls
        for i, (x, y, r, circ) in enumerate(detected_balls):
            if not matched_detected[i]:
                self._add_new_ball(x, y, r, circ)

        # Clean up old balls
        self.tracked_balls = [ball for ball in self.tracked_balls if ball["active"] or ball["ttl"] > -5]

    def _add_new_ball(self, x, y, r, circ):
        """Add a new tracked ball."""
        new_ball = {
            "id": self.next_id,
            "center": (x, y),
            "radius": r,
            "ttl": self.tracking_history,
            "active": True,
            "detection_count": 1,
            "confirmed": False,
            "circularity": circ
        }
        self.tracked_balls.append(new_ball)
        self.track_paths[self.next_id] = deque(maxlen=self.path_length)
        self.track_paths[self.next_id].append((x, y))
        self.next_id += 1

    def _update_tracked_ball(self, tracked_idx, detection):
        """Update an existing tracked ball."""
        tracked_ball = self.tracked_balls[tracked_idx]
        x, y, r, circ = detection

        tracked_ball["center"] = (x, y)
        tracked_ball["radius"] = r
        tracked_ball["ttl"] = self.tracking_history
        tracked_ball["detection_count"] += 1
        tracked_ball["circularity"] = circ

        # Update path
        self.track_paths[tracked_ball["id"]].append((x, y))

        # Confirm ball after enough detections
        if tracked_ball["detection_count"] >= self.min_detection_confidence:
            tracked_ball["confirmed"] = True

    def _check_ball_crossing(self, ball, frame_height):
        """Check if a ball has crossed the dividing line."""
        ball_id = ball["id"]
        current_y = ball["center"][1]
        dividing_y = int(frame_height * self.dividing_line_position)

        # Initialize ball history
        if ball_id not in self.ball_crossing_history:
            if current_y < dividing_y:
                self.ball_crossing_history[ball_id] = "top"
            else:
                self.ball_crossing_history[ball_id] = "bottom"
            return

        # Check cooldown
        if ball_id in self.crossing_cooldown:
            self.crossing_cooldown[ball_id] -= 1
            if self.crossing_cooldown[ball_id] <= 0:
                del self.crossing_cooldown[ball_id]
            return

        previous_side = self.ball_crossing_history[ball_id]

        # Determine current side
        if current_y < dividing_y - 10:
            current_side = "top"
        elif current_y > dividing_y + 10:
            current_side = "bottom"
        else:
            return

        # Check if crossed
        if previous_side != current_side:
            if previous_side == "top" and current_side == "bottom":
                self.total_balls_left -= 1
                self.total_balls_right += 1
                print(f"Ball #{ball_id} crossed TOP -> BOTTOM. Top: {self.total_balls_left}, Bottom: {self.total_balls_right}")
            elif previous_side == "bottom" and current_side == "top":
                self.total_balls_right -= 1
                self.total_balls_left += 1
                print(f"Ball #{ball_id} crossed BOTTOM -> TOP. Top: {self.total_balls_left}, Bottom: {self.total_balls_right}")

            self.ball_crossing_history[ball_id] = current_side
            self.crossing_cooldown[ball_id] = self.cooldown_frames

    def draw_results(self, frame, show_debug=False):
        """Draw tracking results on the frame."""
        frame_height, frame_width = frame.shape[:2]

        # Draw dividing line
        dividing_y = int(frame_height * self.dividing_line_position)
        cv2.line(frame, (0, dividing_y), (frame_width, dividing_y), (0, 255, 255), 3)

        # Display ball counts
        cv2.putText(frame, f"TOP SIDE: {self.total_balls_left}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f"BOTTOM SIDE: {self.total_balls_right}", (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Check for crossings
        for ball in self.tracked_balls:
            if ball["active"] and ball["confirmed"]:
                self._check_ball_crossing(ball, frame_height)

        # Draw balls
        for ball in self.tracked_balls:
            if ball["active"]:
                color = (0, 255, 0) if ball["confirmed"] else (255, 0, 0)
                cv2.circle(frame, ball["center"], self.expected_ball_radius, color, 2)
                cv2.circle(frame, ball["center"], 2, (0, 0, 255), -1)
                cv2.putText(frame, f"#{ball['id']}",
                           (ball["center"][0] - 10, ball["center"][1] - self.expected_ball_radius - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Draw path
                if ball["id"] in self.track_paths and len(self.track_paths[ball["id"]]) > 1:
                    path = list(self.track_paths[ball["id"]])
                    for i in range(1, len(path)):
                        cv2.line(frame, path[i - 1], path[i], color, 2)

        return frame

    def setup_camera(self, cap):
        """Setup camera for fast moving objects - high FPS, fast shutter speed"""
        if not cap.isOpened():
            raise Exception("Could not open camera")

        # Set codec to MJPG for better quality and performance
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        # Set resolution first (common resolutions that most cameras support well)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Set maximum FPS (30 fps as specified)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Try to get better buffer management
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Let camera use auto settings first (like Windows Camera does)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure enabled initially
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto white balance

        # Basic settings that usually work well
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)     # Neutral brightness
        cap.set(cv2.CAP_PROP_CONTRAST, 32)      # Moderate contrast
        cap.set(cv2.CAP_PROP_SATURATION, 64)    # Good saturation for color detection

        print("=== Camera Settings ===")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print("Camera setup successful!")


def create_hsv_sliders():
    """Create trackbars for HSV adjustment."""
    cv2.namedWindow('HSV Controls')
    cv2.resizeWindow('HSV Controls', 400, 300)

    cv2.createTrackbar('Hue Min', 'HSV Controls', DEFAULT_HUE_MIN, 179, lambda x: None)
    cv2.createTrackbar('Hue Max', 'HSV Controls', DEFAULT_HUE_MAX, 179, lambda x: None)
    cv2.createTrackbar('Sat Min', 'HSV Controls', DEFAULT_SAT_MIN, 255, lambda x: None)
    cv2.createTrackbar('Sat Max', 'HSV Controls', DEFAULT_SAT_MAX, 255, lambda x: None)
    cv2.createTrackbar('Val Min', 'HSV Controls', DEFAULT_VAL_MIN, 255, lambda x: None)
    cv2.createTrackbar('Val Max', 'HSV Controls', DEFAULT_VAL_MAX, 255, lambda x: None)


def get_hsv_values():
    """Get current HSV values from trackbars."""
    try:
        hue_min = cv2.getTrackbarPos('Hue Min', 'HSV Controls')
        hue_max = cv2.getTrackbarPos('Hue Max', 'HSV Controls')
        sat_min = cv2.getTrackbarPos('Sat Min', 'HSV Controls')
        sat_max = cv2.getTrackbarPos('Sat Max', 'HSV Controls')
        val_min = cv2.getTrackbarPos('Val Min', 'HSV Controls')
        val_max = cv2.getTrackbarPos('Val Max', 'HSV Controls')
        return (hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    except:
        return (DEFAULT_HUE_MIN, DEFAULT_HUE_MAX, DEFAULT_SAT_MIN, DEFAULT_SAT_MAX, DEFAULT_VAL_MIN, DEFAULT_VAL_MAX)


def main():
    """Main function to run the ping pong ball tracker."""
    parser = argparse.ArgumentParser(description='Fast Ping Pong Ball Tracker')
    parser.add_argument('--camera', type=int, default=DEFAULT_CAMERA_INDEX, help='Camera index')
    parser.add_argument('--hue_min', type=int, default=DEFAULT_HUE_MIN, help='Minimum Hue (0-179)')
    parser.add_argument('--hue_max', type=int, default=DEFAULT_HUE_MAX, help='Maximum Hue (0-179)')
    parser.add_argument('--sat_min', type=int, default=DEFAULT_SAT_MIN, help='Minimum Saturation (0-255)')
    parser.add_argument('--sat_max', type=int, default=DEFAULT_SAT_MAX, help='Maximum Saturation (0-255)')
    parser.add_argument('--val_min', type=int, default=DEFAULT_VAL_MIN, help='Minimum Value (0-255)')
    parser.add_argument('--val_max', type=int, default=DEFAULT_VAL_MAX, help='Maximum Value (0-255)')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    parser.add_argument('--sliders', action='store_true', help='Show HSV adjustment sliders')
    parser.add_argument('--dividing_line', type=float, default=0.5, help='Dividing line position (0.0-1.0)')

    args = parser.parse_args()

    # Set up HSV color range
    ball_color_lower = [args.hue_min, args.sat_min, args.val_min]
    ball_color_upper = [args.hue_max, args.sat_max, args.val_max]

    print(f"Using HSV range: {ball_color_lower} to {ball_color_upper}")

    # Initialize tracker
    tracker = PingPongBallTracker(ball_color_lower, ball_color_upper, args.dividing_line)

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # Setup camera
    tracker.setup_camera(cap)

    print("Fast Ping Pong Ball Tracker Started!")
    print("Controls: ESC/'q' - Quit, 'd' - Debug, 's' - Sliders, 'r' - Reset, SPACE - Pause")

    paused = False
    debug_mode = args.debug
    sliders_mode = args.sliders

    if sliders_mode:
        create_hsv_sliders()

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break

                # Flip frame
                frame = cv2.flip(frame, -1)

                # Update HSV values from sliders if enabled
                if sliders_mode:
                    hue_min, hue_max, sat_min, sat_max, val_min, val_max = get_hsv_values()
                    tracker.update_hsv_range(hue_min, hue_max, sat_min, sat_max, val_min, val_max)

                # Detect balls
                detected_balls, mask = tracker.detect_balls(frame)

                # Update tracking
                tracker.update_tracking(detected_balls)

                # Draw results
                result_frame = tracker.draw_results(frame.copy(), debug_mode)

                # Show debug mask if enabled
                if debug_mode:
                    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    mask_resized = cv2.resize(mask_colored, (200, 150))
                    h, w = result_frame.shape[:2]
                    result_frame[h - 160:h - 10, w - 210:w - 10] = mask_resized

                cv2.imshow('Fast Ping Pong Ball Tracker', result_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('d'):  # Toggle debug mode
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('s'):  # Toggle sliders mode
                sliders_mode = not sliders_mode
                if sliders_mode:
                    create_hsv_sliders()
                    print("HSV sliders: ON")
                else:
                    try:
                        cv2.destroyWindow('HSV Controls')
                    except:
                        pass
                    print("HSV sliders: OFF")
            elif key == ord('r'):  # Reset tracking
                tracker.tracked_balls = []
                tracker.track_paths = {}
                tracker.next_id = 0
                print("Tracking reset")
            elif key == ord(' '):  # Space to pause/resume
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Fast Ping Pong Ball Tracker stopped")


if __name__ == "__main__":
    main()
