import cv2
import pandas as pd
import os
import argparse
import numpy as np

# Define colors in BGR format for OpenCV
COLOR_RIGHT = (83, 50, 250)   # #fa3253 (pink) in BGR
COLOR_LEFT = (255, 221, 51)   # #33ddff (cyan) in BGR
COLOR_OTHER = (102, 255, 102) # #66ff66 (green) in BGR


def load_annotations(csv_path):
    """Load CSV annotations."""
    return pd.read_csv(csv_path)

def find_image(image_name, root_dir):
    """Recursively search for an image file in the root directory."""
    for dirpath, _, filenames in os.walk(root_dir):
        if image_name in filenames:
            return os.path.join(dirpath, image_name)
    return None

def draw_annotations(image, row, show_bbox, show_kpts, show_text):
    """Draw keypoints and bounding box on the image."""
    h, w, _ = image.shape
    
    # Draw bounding box if enabled
    if show_bbox:
        if not any(pd.isna([row['bbox_tl-x'], row['bbox_tl-y'], row['bbox_br-x'], row['bbox_br-y']])):
            tl_x, tl_y, br_x, br_y = int(row['bbox_tl-x']), int(row['bbox_tl-y']), int(row['bbox_br-x']), int(row['bbox_br-y'])
            cv2.rectangle(image, (tl_x, tl_y), (br_x, br_y), (0, 255, 255), 2)
    
    # Draw keypoints
    for col in row.index:
        if '-x' in col:
            keypoint = col[:-2]  # Remove '-x'
            if pd.isna(row[f'{keypoint}-x']) or pd.isna(row[f'{keypoint}-y']):
                continue  # Skip keypoints with missing coordinates
            x, y = int(row[f'{keypoint}-x']), int(row[f'{keypoint}-y'])
            
            if 'bbox' in keypoint.lower():
                continue  # Skip bounding box keypoints

            if 'right' in keypoint.lower():
                color = COLOR_RIGHT
            elif 'left' in keypoint.lower():
                color = COLOR_LEFT
            else:
                color = COLOR_OTHER
            
            if show_kpts:
                cv2.circle(image, (x, y), 5, color, -1)
            if show_text:
                cv2.putText(image, keypoint, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Draw filename in the top center
    text_size = cv2.getTextSize(row['filename'], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(image, row['filename'], (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (  0,   0,   0), 5, cv2.LINE_AA)
    cv2.putText(image, row['filename'], (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def crop_and_zoom(image, row):
    """Crop the image around the bounding box and enlarge it 2x with keypoints."""
    if any(pd.isna([row['bbox_tl-x'], row['bbox_tl-y'], row['bbox_br-x'], row['bbox_br-y']])):
        return None
    
    tl_x, tl_y, br_x, br_y = int(row['bbox_tl-x']), int(row['bbox_tl-y']), int(row['bbox_br-x']), int(row['bbox_br-y'])
    cropped = image[tl_y:br_y, tl_x:br_x]
    cropped = cv2.resize(cropped, (cropped.shape[1] * 3, cropped.shape[0] * 3))
    
    scale_x = 3
    scale_y = 3
    offset_x = -tl_x * scale_x
    offset_y = -tl_y * scale_y
    
    for col in row.index:
        if '-x' in col:
            keypoint = col[:-2]
            if pd.isna(row[f'{keypoint}-x']) or pd.isna(row[f'{keypoint}-y']):
                continue
            x = int(row[f'{keypoint}-x'] * scale_x + offset_x)
            y = int(row[f'{keypoint}-y'] * scale_y + offset_y)
            
            if 'bbox' in keypoint.lower():
                continue  # Skip bounding box keypoints

            if 'right' in keypoint.lower():
                color = COLOR_RIGHT
            elif 'left' in keypoint.lower():
                color = COLOR_LEFT
            else:
                color = COLOR_OTHER
            
            cv2.circle(cropped, (x, y), 5, color, -1)
            cv2.putText(cropped, keypoint, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return cropped

def main(csv_path, image_dir):
    df = load_annotations(csv_path)
    index = 0
    show_bbox = True
    show_kpts = True
    show_text = True
    
    while index < len(df):
        row = df.iloc[index]
        img_path = find_image(row['filename'], image_dir)
        
        if img_path is None:
            print(f"Image not found: {row['filename']}")
            index += 1
            continue
        
        original_image = cv2.imread(img_path)
        image = original_image.copy()
        draw_annotations(image, row, show_bbox, show_kpts, show_text)
        
        # Add legend
        legend = ["-> / D: Next image", 
            "<- / A: Previous image", \
            "B: Toggle bbox", 
            "K: Toggle keypoints", 
            "T: Toggle text", 
            "C: Crop & Zoom",
            "ESC: Exit"]
        y_offset = 20
        for text in legend:
            cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        cv2.imshow("Annotations", image)
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  # ESC to exit
            break
        elif key == ord('c'):
            cropped = crop_and_zoom(original_image, row)
            if cropped is not None:
                cv2.imshow("Cropped & Zoomed", cropped)
                while True:
                    crop_key = cv2.waitKey(10) & 0xFF
                    if crop_key == 27:
                        cv2.destroyWindow("Cropped & Zoomed")
                        break
                    elif cv2.getWindowProperty("Cropped & Zoomed", cv2.WND_PROP_VISIBLE) < 1:
                        break
        elif key == ord('b'):  # B key
            show_bbox = not show_bbox
        elif key == ord('k'):  # K key
            show_kpts = not show_kpts
        elif key == ord('t'):  # T key
            show_text = not show_text
        elif key == 83 or key == ord('d'):  # Right arrow or 'D' key
            index = min(index + 1, len(df) - 1)
        elif key == 81 or key == ord('a'):  # Left arrow or 'A' key
            index = max(index - 1, 0)
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check labeled images with annotations.")
    parser.add_argument("csv_path", type=str, help="Path to the annotation CSV file.")
    parser.add_argument("image_dir", type=str, help="Root directory containing images.")
    args = parser.parse_args()
    main(args.csv_path, args.image_dir)
