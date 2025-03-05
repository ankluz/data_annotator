# Data Annotator

A Python tool for annotating bounding boxes and keypoints in images. This tool provides functionality for detecting, drawing, and annotating bounding boxes around characters or objects in images, as well as adding keypoint sequences within these boxes.

## Usage

```python
from data_annotator import annotate_keypoints

annotate_keypoints('path/to/image.jpg', 'path/to/output.json')
```

# Detect boxes automatically
```python
boxes = detect_and_draw_boxes("image.jpg", "output.jpg")
```

# Manually adjust boxes
```python
annotations = manual_box_adjustment("image.jpg", boxes)
```

# Add keypoint annotations
```python
annotated_data = annotate_keypoints("image.jpg", annotations)
```

# Save annotations
```python
save_annotations(annotated_data, "annotations.json")
```

# Load annotations
```python
loaded_data = load_annotations("annotations.json")
```

## Features

- Automatic bounding box detection for characters/objects
- Interactive keypoint annotation within bounding boxes
- Support for multiple keypoint sequences per box
- Zoom and pan functionality for precise annotation
- Character/symbol labeling for each bounding box
- JSON-based annotation storage
- Help overlay with keyboard shortcuts

## Controls

- **Left Click**: Add keypoint
- **N**: Start new sequence
- **E**: Next sequence
- **Q**: Previous sequence
- **Z**: Undo last keypoint
- **D**: Next box
- **A**: Previous box
- **T**: Enter symbol/character
- **Mouse Wheel**: Zoom in/out
- **Middle Mouse/Shift+Drag**: Pan image
- **S**: Save annotations
- **R**: Reset view
- **H**: Toggle help
- **ESC**: Exit


## Requirements

- OpenCV
- NumPy
- Matplotlib
- keyboard

