import cv2
import numpy as np
from data_annotator.help_overlay import HelpOverlay
import keyboard
from data_annotator.file_processor import save_annotations

def manual_box_adjustment(image_path, bounding_boxes):
    """
    Interactive tool for manual adjustment of bounding boxes.
    
    Args:
        image_path: Path to the image
        bounding_boxes: List of tuples (box, keypoints) or (box, keypoints, symbol) or list of boxes (x,y,w,h)
        
    Returns:
        list: Adjusted bounding boxes in format [(box, sequences, symbol), ...]
    """
    # Load image and check for errors
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return []
    
    height, width = image.shape[:2]
    
    # Initialize view parameters
    zoom_scale = 1.0
    pan_offset = [0, 0]
    
    # Initialize interaction states
    dragging = False
    resizing = False
    creating_box = False
    panning = False
    selected_box = None
    selected_boxes = set()
    drag_start = None
    resize_handle = None
    creation_start = None
    last_pan_pos = None
    box_start = None
    resize_start = None
    
    # Calculate display scale
    display_scale = min(1440 / width, 900 / height)
    display_width = int(width * display_scale)
    display_height = int(height * display_scale)
    
    def transform_to_display_coords(box):
        """Transform box coordinates from image space to display space"""
        x, y, w, h = box
        return (
            int(x * display_scale),
            int(y * display_scale),
            int(w * display_scale),
            int(h * display_scale)
        )
    
    def transform_to_image_coords(box):
        """Transform box coordinates from display space to image space"""
        x, y, w, h = box
        return (
            x / display_scale,
            y / display_scale,
            w / display_scale,
            h / display_scale
        )

    # Ensure data compatibility and proper format
    if not bounding_boxes:
        annotations = []
        boxes = []
    elif isinstance(bounding_boxes[0], tuple):
        if len(bounding_boxes[0]) == 3:  # New format (box, sequences, symbol)
            annotations = bounding_boxes
            boxes = [box for box, _, _ in annotations]
        elif len(bounding_boxes[0]) == 2:  # Old format (box, sequences)
            annotations = [(box, sequences, "") for box, sequences in bounding_boxes]
            boxes = [box for box, _, _ in annotations]
        else:
            boxes = [bounding_boxes[0]]
            annotations = [(box, [], "") for box in boxes]
    else:
        boxes = [tuple(float(x) for x in box) for box in bounding_boxes]
        annotations = [(box, [], "") for box in boxes]

    running = True
    show_help = False
    
    # Изменяем масштабирование изображения и координат
    max_height = 900
    max_width = 1600
    height, width = image.shape[:2]
    scale = min(max_width/width, max_height/height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = cv2.resize(image, (new_width, new_height))
    
    # Сохраняем оригинальные координаты без масштабирования
    original_boxes = boxes.copy()
    
    # Масштабируем boxes только для отображения
    boxes = [(float(x*scale), float(y*scale), float(w*scale), float(h*scale)) 
             for (x, y, w, h) in boxes]
    
    # Initialize state variables
    selected_box = None
    dragging = False
    drag_start = None
    box_start = None
    selected_boxes = set()
    panning = False
    last_pan_pos = None
    resizing = False
    resize_handle = None
    resize_start = None
    
    # Add new variables for box creation
    creating_box = False
    creation_start = None
    
    global mouse_x, mouse_y, real_x, real_y 
    mouse_x, mouse_y = 0, 0
    real_x, real_y = 0, 0
    
    # Initialize undo/redo stacks
    undo_stack = []
    redo_stack = []
    
    def save_state():
        """Save current state for undo"""
        undo_stack.append([box[:] for box in boxes])
        redo_stack.clear()
    
    def undo():
        """Undo last action"""
        if undo_stack:
            redo_stack.append([box[:] for box in boxes])
            boxes[:] = undo_stack.pop()
            return True
        return False
    
    def redo():
        """Redo last undone action"""
        if redo_stack:
            undo_stack.append([box[:] for box in boxes])
            boxes[:] = redo_stack.pop()
            return True
        return False
    
    def apply_zoom_and_pan(x, y):
        """Transform coordinates according to zoom and pan"""
        x = (x - pan_offset[0]) / zoom_scale
        y = (y - pan_offset[1]) / zoom_scale
        return int(x), int(y)
    
    def reset_view():
        """Reset zoom and pan to default values"""
        nonlocal zoom_scale, pan_offset
        zoom_scale = 1.0
        pan_offset = [0, 0]
    
    def constrain_pan_offset():
        """Constrain pan offset to keep image partially visible"""
        nonlocal pan_offset
        visible_width = image.shape[1] * zoom_scale
        visible_height = image.shape[0] * zoom_scale
        min_visible = 0.25
        
        min_x = -visible_width * (1 - min_visible)
        max_x = new_width - visible_width * min_visible
        min_y = -visible_height * (1 - min_visible)
        max_y = new_height - visible_height * min_visible
        
        pan_offset[0] = max(min_x, min(max_x, pan_offset[0]))
        pan_offset[1] = max(min_y, min(max_y, pan_offset[1]))
    
    def get_resize_handle(x, y, box):
        """Determine which resize handle (if any) is under the cursor"""
        bx, by, bw, bh = box
        sx = int(bx * zoom_scale + pan_offset[0])
        sy = int(by * zoom_scale + pan_offset[1])
        sw = int(bw * zoom_scale)
        sh = int(bh * zoom_scale)
        
        handle_size = 10
        
        # Check corners
        if abs(x - sx) < handle_size and abs(y - sy) < handle_size:
            return 'topleft'
        if abs(x - (sx + sw)) < handle_size and abs(y - sy) < handle_size:
            return 'topright'
        if abs(x - sx) < handle_size and abs(y - (sy + sh)) < handle_size:
            return 'bottomleft'
        if abs(x - (sx + sw)) < handle_size and abs(y - (sy + sh)) < handle_size:
            return 'bottomright'
            
        # Check sides
        if abs(y - sy) < handle_size and sx < x < sx + sw:
            return 'top'
        if abs(y - (sy + sh)) < handle_size and sx < x < sx + sw:
            return 'bottom'
        if abs(x - sx) < handle_size and sy < y < sy + sh:
            return 'left'
        if abs(x - (sx + sw)) < handle_size and sy < y < sy + sh:
            return 'right'
            
        return None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal dragging, resizing, selected_box, selected_boxes
        nonlocal drag_start, resize_handle, creation_start, creating_box
        nonlocal zoom_scale, pan_offset, boxes, annotations
        nonlocal panning, last_pan_pos, box_start, resize_start
        global mouse_x, mouse_y, real_x, real_y
        
        mouse_x, mouse_y = x, y
        real_x = (x - pan_offset[0]) / zoom_scale
        real_y = (y - pan_offset[1]) / zoom_scale
        
        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_direction = flags >> 16
            old_zoom = zoom_scale
            
            # Smoother zooming
            zoom_factor = 1.05
            if wheel_direction > 0:
                zoom_scale *= zoom_factor
            else:
                zoom_scale /= zoom_factor
            zoom_scale = max(0.1, min(5.0, zoom_scale))
            
            # Zoom relative to screen center
            center_x = display_width / 2
            center_y = display_height / 2
            
            # Update offset relative to center
            pan_offset[0] = center_x - (center_x - pan_offset[0]) * (zoom_scale / old_zoom)
            pan_offset[1] = center_y - (center_y - pan_offset[1]) * (zoom_scale / old_zoom)
            
            constrain_pan_offset()
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            if keyboard.is_pressed('alt'):  # Alt key is pressed
                creating_box = True
                creation_start = (real_x, real_y)
                selected_box = None
                selected_boxes.clear()
                return
            
            save_state()  # Save state before modification
            found = False
            for i, (bx, by, bw, bh) in enumerate(boxes):
                # First check if we're clicking a resize handle
                if i == selected_box:
                    resize_handle = get_resize_handle(x, y, boxes[i])
                    if resize_handle:
                        resizing = True
                        resize_start = (real_x, real_y)
                        box_start = boxes[i]
                        found = True
                        break
                
                # Then check if we're clicking inside the box
                if bx <= real_x <= bx + bw and by <= real_y <= by + bh:
                    selected_box = i
                    selected_boxes = {i}
                    dragging = True
                    drag_start = (real_x, real_y)
                    box_start = (bx, by)
                    found = True
                    break
                    
            if not found:
                selected_box = None
                selected_boxes.clear()
                dragging = False
                resizing = False
            
        elif event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and keyboard.is_pressed('shift')):
            panning = True
            last_pan_pos = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if creating_box and creation_start:
                # Just update the display, actual box will be created on mouse up
                pass
            elif panning and last_pan_pos:
                dx = x - last_pan_pos[0]
                dy = y - last_pan_pos[1]
                pan_offset[0] += dx
                pan_offset[1] += dy
                constrain_pan_offset()
                last_pan_pos = (x, y)
            elif resizing and selected_box is not None:
                # Get current box coordinates
                bx, by, bw, bh = box_start
                # Calculate delta from resize start position
                dx = real_x - resize_start[0]
                dy = real_y - resize_start[1]
                
                # Update box dimensions based on resize handle
                if 'left' in resize_handle:
                    new_w = max(5, bw - dx)
                    new_x = bx + (bw - new_w)
                    bw = new_w
                    bx = new_x
                elif 'right' in resize_handle:
                    bw = max(5, bw + dx)
                    
                if 'top' in resize_handle:
                    new_h = max(5, bh - dy)
                    new_y = by + (bh - new_h)
                    bh = new_h
                    by = new_y
                elif 'bottom' in resize_handle:
                    bh = max(5, bh + dy)
                    
                boxes[selected_box] = (bx, by, bw, bh)
            elif dragging and selected_box is not None:
                # Move box with drag & drop
                dx = real_x - drag_start[0]
                dy = real_y - drag_start[1]
                bx = box_start[0] + dx
                by = box_start[1] + dy
                _, _, w, h = boxes[selected_box]
                boxes[selected_box] = (bx, by, w, h)
                
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            if creating_box and creation_start:
                # Calculate box dimensions in real image coordinates
                end_x = real_x
                end_y = real_y
                
                # Ensure positive width and height
                start_x = min(creation_start[0], end_x)
                start_y = min(creation_start[1], end_y)
                width = abs(end_x - creation_start[0])
                height = abs(end_y - creation_start[1])
                
                # Add new box if it's large enough
                if width > 5 and height > 5:
                    save_state()  # Save state before adding new box
                    new_box = (start_x, start_y, width, height)
                    boxes.append(new_box)
                    annotations.append((new_box, [], ""))
                    selected_box = len(boxes) - 1
                    selected_boxes = {selected_box}
                
                creating_box = False
                creation_start = None
            
            dragging = False
            resizing = False
            resize_handle = None
            panning = False
            last_pan_pos = None

    window_name = "Adjust Bounding Boxes"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    help_overlay = HelpOverlay()
    
    # Add keyboard handlers before the main loop
    def on_key_press(event):
        nonlocal zoom_scale, pan_offset, selected_boxes, selected_box, boxes, running
        nonlocal show_help, annotations
        
        scan_code_mapping = {
            35: 'h',
            19: 'r',
            45: 'x',
            31: 's',
            22: 'u',
            24: 'o',
            25: 'p',
            51: ',',
            52: '.',
            44: 'z',
            21: 'y',
        }
        
        actual_key = scan_code_mapping.get(event.scan_code, event.name)
        
        # Handle Ctrl+Z and Ctrl+Y
        if keyboard.is_pressed('ctrl'):
            if actual_key == 'z':
                if undo():
                    print("Undo performed")
                return
            elif actual_key == 'y':
                if redo():
                    print("Redo performed")
                return
        
        if actual_key == 'h':
            help_overlay.toggle()
            print("Help toggled")
        elif actual_key == 'r':
            reset_view()
        elif actual_key == 'x' or actual_key == 'delete' or actual_key == 'backspace':
            if selected_boxes:
                for box_idx in sorted(list(selected_boxes), reverse=True):
                    if 0 <= box_idx < len(boxes):
                        boxes.pop(box_idx)
                selected_boxes.clear()
                selected_box = None
        elif actual_key == 's':
            annotations = [(box, sequences, symbol) for box, (sequences, symbol) in zip(boxes, annotations)]
            save_annotations(annotations, image_path + '.json')
            print(f"Boxes saved to: {image_path}.json")
        elif actual_key == 'esc':
            running = False
        elif actual_key == 'up':
            if selected_box is not None:
                x, y, w, h = boxes[selected_box]
                boxes[selected_box] = (x, y - 1, w, h)
        elif actual_key == 'down':
            if selected_box is not None:
                x, y, w, h = boxes[selected_box]
                boxes[selected_box] = (x, y + 1, w, h)
        elif actual_key == 'left':
            if selected_box is not None:
                x, y, w, h = boxes[selected_box]
                boxes[selected_box] = (x - 1, y, w, h)
        elif actual_key == 'right':
            if selected_box is not None:
                x, y, w, h = boxes[selected_box]
                boxes[selected_box] = (x + 1, y, w, h)

    # Register keyboard handlers for specific keys
    keys_to_monitor = [
        'h', 'r', 'x', 'delete', 'backspace', 's', 'esc', 'z', 'y',
        'up', 'down', 'left', 'right',
        'alt', 'shift', 'ctrl'
    ]
    
    for key in keys_to_monitor:
        keyboard.on_press_key(key, on_key_press, suppress=True)

    while running:
        # Apply zoom and pan to image
        M = np.float32([[zoom_scale, 0, pan_offset[0]], 
                       [0, zoom_scale, pan_offset[1]]])
        img_copy = cv2.warpAffine(image.copy(), M, (display_width, display_height))
        
        # Draw all boxes
        for i, (x, y, w, h) in enumerate(boxes):
            sx = int(x * zoom_scale + pan_offset[0])
            sy = int(y * zoom_scale + pan_offset[1])
            sw = int(w * zoom_scale)
            sh = int(h * zoom_scale)
            
            color = (0, 255, 0) if i in selected_boxes else (0, 0, 255)
            cv2.rectangle(img_copy, (sx, sy), (sx+sw, sy+sh), color, 2)
            
            # Draw resize handles for selected box
            if i == selected_box:
                handle_size = 4
                cv2.circle(img_copy, (sx, sy), handle_size, color, -1)
                cv2.circle(img_copy, (sx + sw, sy), handle_size, color, -1)
                cv2.circle(img_copy, (sx, sy + sh), handle_size, color, -1)
                cv2.circle(img_copy, (sx + sw, sy + sh), handle_size, color, -1)
                cv2.circle(img_copy, (sx + sw//2, sy), handle_size, color, -1)
                cv2.circle(img_copy, (sx + sw//2, sy + sh), handle_size, color, -1)
                cv2.circle(img_copy, (sx, sy + sh//2), handle_size, color, -1)
                cv2.circle(img_copy, (sx + sw, sy + sh//2), handle_size, color, -1)
        
        # Draw box being created
        if creating_box and creation_start:
            start_x = int(creation_start[0] * zoom_scale + pan_offset[0])
            start_y = int(creation_start[1] * zoom_scale + pan_offset[1])
            curr_x = int(real_x * zoom_scale + pan_offset[0])
            curr_y = int(real_y * zoom_scale + pan_offset[1])
            cv2.rectangle(img_copy, (start_x, start_y), (curr_x, curr_y), (0, 255, 0), 2)
        
        img_copy = help_overlay.draw(img_copy)
        
        cv2.imshow(window_name, img_copy)
        if cv2.waitKey(1) == 27:  # ESC key
            running = False

    # Cleanup
    keyboard.unhook_all()
    cv2.destroyAllWindows()
    
    # Return the annotations with original coordinates
    final_annotations = []
    for i, box in enumerate(boxes):
        # Convert display coordinates back to original image coordinates
        original_box = (box[0]/scale, box[1]/scale, box[2]/scale, box[3]/scale)
        # Get the sequences and symbol for this box from annotations
        sequences = annotations[i][1] if i < len(annotations) else []
        symbol = annotations[i][2] if i < len(annotations) else ""
        final_annotations.append((original_box, sequences, symbol))
    
    return final_annotations
