#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import os
import datetime
import logging
import psutil
import threading

class ResourceMonitor(threading.Thread):
    """Monitor system resources in a separate thread"""
    def __init__(self, interval=1.0, log_file=None):
        threading.Thread.__init__(self)
        self.interval = interval
        self.running = True
        self.daemon = True  # Thread will exit when main program exits
        self.cpu_usage = []
        self.ram_usage = []
        self.log_file = log_file
        
    def run(self):
        while self.running:
            # Get CPU and RAM usage
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_percent = psutil.virtual_memory().percent
            
            # Store readings
            self.cpu_usage.append(cpu_percent)
            self.ram_usage.append(ram_percent)
            
            # Log resource usage
            if self.log_file:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.log_file, 'a') as f:
                    f.write(f"[{timestamp}] CPU: {cpu_percent:.1f}%, RAM: {ram_percent:.1f}%\n")
            
            # Sleep for the specified interval
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
    
    def get_average_usage(self):
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_ram = sum(self.ram_usage) / len(self.ram_usage) if self.ram_usage else 0
        return avg_cpu, avg_ram

def setup_logging(log_dir="/app/logs"):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging
    log_file = os.path.join(log_dir, f"detection_{timestamp}.log")
    resource_log_file = os.path.join(log_dir, f"resources_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger('people_detector')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, resource_log_file

def detect_people(source, output_dir="/app/outputs", log_dir="/app/logs"):
    """Enhanced people detection function with logging and resource monitoring"""
    # Set up logging
    logger, resource_log_file = setup_logging(log_dir)
    logger.info("Starting people detection")
    
    # Start resource monitoring
    logger.info("Starting resource monitoring")
    monitor = ResourceMonitor(interval=2.0, log_file=resource_log_file)
    monitor.start()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize the HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    logger.info("Initialized HOG detector")
    
    # Open video source
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        source_name = f"Camera {source}"
    else:
        cap = cv2.VideoCapture(source)
        source_name = os.path.basename(source)
    
    if not cap.isOpened():
        logger.error(f"Could not open {source_name}")
        monitor.stop()
        return
    
    logger.info(f"Running detection on {source_name}")
    print("Press ESC to exit")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"detected_{timestamp}.avi")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    logger.info(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # For calculating FPS and stats
    frame_count = 0
    processed_count = 0
    detection_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_update_interval = 10  # Update FPS every 10 frames
    detection_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream")
                break
            
            frame_count += 1
            
            # Process every 3rd frame for better performance
            process_this_frame = (frame_count % 3 == 0)
            if process_this_frame:
                processed_count += 1
                detection_start = time.time()
                
                # Resize for faster detection
                small_frame = cv2.resize(frame, (640, 480))
                
                # Detect people
                boxes, weights = hog.detectMultiScale(
                    small_frame, 
                    winStride=(8, 8), 
                    padding=(4, 4), 
                    scale=1.05
                )
                
                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
                
                # Scale coordinates to original frame size
                scaled_boxes = []
                if len(boxes) > 0:
                    for i, (x, y, w, h) in enumerate(boxes):
                        confidence = float(weights[i])
                        if confidence > 0.3:  # Simple confidence threshold
                            x_orig = int(x * (width / 640))
                            y_orig = int(y * (height / 480))
                            w_orig = int(w * (width / 640))
                            h_orig = int(h * (height / 480))
                            scaled_boxes.append((x_orig, y_orig, w_orig, h_orig, confidence))
                            
                # Log detections
                current_detection_count = len(scaled_boxes)
                detection_count += current_detection_count
                
                if current_detection_count > 0:
                    logger.info(f"Frame {frame_count}: Detected {current_detection_count} people")
                
                # Draw boxes on frame
                for i, (x, y, w, h, confidence) in enumerate(scaled_boxes):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Person {i+1}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate and display FPS
            if frame_count % fps_update_interval == 0:
                elapsed_time = time.time() - fps_start_time
                current_fps = fps_update_interval / elapsed_time if elapsed_time > 0 else 0
                fps_start_time = time.time()
                
                # Add resource usage to the frame
                cpu_usage, ram_usage = monitor.get_average_usage()
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"CPU: {cpu_usage:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"RAM: {ram_usage:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Count people
            person_count = len(scaled_boxes) if 'scaled_boxes' in locals() and process_this_frame else 0
            cv2.putText(frame, f"People: {person_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (width - 230, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow("People Detection", frame)
            
            # Write to output file
            out.write(frame)
            
            # Check for ESC key
            if cv2.waitKey(1) == 27:
                logger.info("Detection stopped by user")
                break
                
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
    finally:
        # Stop resource monitoring
        monitor.stop()
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Log summary statistics
        total_time = time.time() - start_time
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        avg_cpu, avg_ram = monitor.get_average_usage()
        
        logger.info(f"Detection completed. Output saved to {output_path}")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Frames analyzed for detection: {processed_count}")
        logger.info(f"Total people detected: {detection_count}")
        logger.info(f"Average detection time: {avg_detection_time*1000:.2f}ms")
        logger.info(f"Total elapsed time: {total_time:.2f}s")
        logger.info(f"Average CPU usage: {avg_cpu:.1f}%")
        logger.info(f"Average RAM usage: {avg_ram:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Enhanced People Detection with Logging and Resource Monitoring")
    
    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--camera", type=int, help="Camera device ID (e.g., 0 for webcam)")
    source_group.add_argument("--video", help="Path to video file")
    
    # Optional paths
    parser.add_argument("--output-dir", default="/app/outputs", help="Directory to save output files")
    parser.add_argument("--log-dir", default="/app/logs", help="Directory to save log files")
    
    args = parser.parse_args()
    
    if args.camera is not None:
        detect_people(args.camera, output_dir=args.output_dir, log_dir=args.log_dir)
    elif args.video:
        detect_people(args.video, output_dir=args.output_dir, log_dir=args.log_dir)

if __name__ == "__main__":
    main()
