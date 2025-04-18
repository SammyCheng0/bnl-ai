import xml.etree.ElementTree as ET

# Load the XML file
file_path = "/Users/marcoschivo/Downloads/20241115-152700-409239_3-1263/annotations.xml"

# Parse XML
tree = ET.parse(file_path)
root = tree.getroot()

# Display the structure
def print_structure(element, level=0):
    indent = "  " * level
    print(f"{indent}- {element.tag}")
    for child in element:
        print_structure(child, level + 1)

print("XML Structure:")
print_structure(root)

# Extract Metadata
meta = root.find("meta")
task = meta.find("task") if meta is not None else None
if task:
    task_name = task.find("name").text
    task_id = task.find("id").text
    created = task.find("created").text
    print(f"\nTask Info:\n - ID: {task_id}\n - Name: {task_name}\n - Created: {created}")

# Extract Labels
labels = task.find("labels") if task is not None else None
if labels:
    print("\nLabels:")
    for label in labels.findall("label"):
        name = label.find("name").text
        label_type = label.find("type").text
        color = label.find("color").text
        print(f" - {name} ({label_type}, Color: {color})")

# Extract Keypoints (Skeleton Data)
print("\nKeypoints in Skeletons:")
for image in root.findall("image"):
    img_name = image.get("name")
    print(f"\nImage: {img_name}")
    
    for skeleton in image.findall("skeleton"):
        label = skeleton.get("label")
        print(f"  Skeleton: {label}")

        for point in skeleton.findall("points"):
            point_label = point.get("label")
            coords = point.get("points")
            print(f"    - {point_label}: {coords}")

# Extract Bounding Box Data
print("\nBounding Boxes:")
for image in root.findall("image"):
    img_name = image.get("name")
    for box in image.findall("box"):
        label = box.get("label")
        xtl, ytl, xbr, ybr = box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")
        print(f" - Image: {img_name}, Label: {label}, Box: ({xtl}, {ytl}) -> ({xbr}, {ybr})")
