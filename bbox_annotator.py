import cv2
import numpy as np
from data_annotator.help_overlay import HelpOverlay
import keyboard  # Add import at the beginning
from data_annotator.file_processor import save_annotations


def annotate_keypoints(image_path, boxes_data, output_path=None):
    """
    Interactive tool for annotating keypoints within bounding boxes.
    
    Args:
        image_path: Path to the image
        boxes_data: List of tuples (box, keypoints_sequences) or (box, keypoints_sequences, symbol) or list of boxes (x,y,w,h)
        output_path: Optional path to save/load annotations
        
    Returns:
        list: List of tuples (box, keypoints_sequences, symbol)
              where box is (x,y,w,h), keypoints_sequences is list of lists of (x,y) points,
              and symbol is a string representing the character in the box
    """
    # Initialize data structure with proper type checking
    if not boxes_data:
        print("Warning: No boxes provided")
        return []
        
    if isinstance(boxes_data[0], tuple):
        if len(boxes_data[0]) == 3:  # Already in new format (box, sequences, symbol)
            annotations = boxes_data
        elif len(boxes_data[0]) == 2:  # Old format (box, sequences)
            annotations = [(box, sequences, "") for box, sequences in boxes_data]
        else:
            annotations = [(box, [], "") for box in boxes_data]
    else:
        print("Warning: Invalid input format")
        return []

    # Проверка на пустой список аннотаций после конвертации
    if not annotations:
        print("Warning: No valid annotations to process")
        return []

    # Load image and calculate display size
    image = cv2.imread(image_path)
    display_width, display_height = 1270, 860
    
    current_box_idx = 0
    current_sequence = []  # Current sequence of keypoints
    all_sequences = annotations[current_box_idx][1]  # All sequences for current box
    current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
    zoom_scale = 1.0
    pan_offset = [0, 0]
    show_help = False
    running = True
    panning = False
    last_pan_pos = None
    entering_symbol = False
    symbol_input = ""

    current_sequence_idx = -1  # Index of current sequence

    help_overlay = HelpOverlay()  # Initialize help overlay
    
    def draw_help(img):
        if not show_help:
            return img
            
        help_text = [
            "Keypoint Annotation Controls:",
            "Left Click - Add keypoint",
            "N - Start new sequence",
            "E - Next sequence",
            "Q - Previous sequence",
            "Z - Undo last keypoint",
            "D - Next box",
            "A - Previous box",
            "T - Enter symbol/character",
            "Mouse Wheel - Zoom in/out",
            "Middle Mouse/Shift+Drag - Pan image",
            "S - Save annotations",
            "R - Reset view",
            "H - Toggle help",
            "ESC - Exit"
        ]
        
        overlay = img.copy()
        padding = 10
        start_x = 20
        start_y = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        
        # Calculate text dimensions
        text_height = 25
        max_width = max([cv2.getTextSize(text, font, font_scale, 1)[0][0] for text in help_text])
        bg_width = max_width + 2 * padding
        bg_height = len(help_text) * text_height + 2 * padding
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, 
                     (start_x - padding, start_y - padding - text_height),
                     (start_x + bg_width, start_y + bg_height),
                     (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        
        # Draw text
        for i, line in enumerate(help_text):
            y = start_y + i * text_height
            cv2.putText(img, line, (start_x, y), font, font_scale, (255, 255, 255), 1)
            
        return img

    def on_mouse(event, x, y, flags, param):
        nonlocal zoom_scale, pan_offset, panning, last_pan_pos, current_sequence
        
        # Don't process mouse events when entering symbol
        if entering_symbol:
            return
            
        # Transform coordinates back to original image space
        orig_x = (x - pan_offset[0]) / zoom_scale
        orig_y = (y - pan_offset[1]) / zoom_scale
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_sequence.append((int(orig_x), int(orig_y)))
        
        elif event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and keyboard.is_pressed('shift')):
            panning = True
            last_pan_pos = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and panning and last_pan_pos:
            dx = x - last_pan_pos[0]
            dy = y - last_pan_pos[1]
            pan_offset[0] += dx
            pan_offset[1] += dy
            last_pan_pos = (x, y)
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            wheel_direction = flags >> 16
            old_zoom = zoom_scale
            
            zoom_factor = 1.1
            if wheel_direction > 0:
                zoom_scale *= zoom_factor
            else:
                zoom_scale /= zoom_factor
            zoom_scale = max(0.1, min(5.0, zoom_scale))
            
            # Zoom relative to center
            center_x = display_width / 2
            center_y = display_height / 2
            pan_offset[0] = center_x - (center_x - pan_offset[0]) * (zoom_scale / old_zoom)
            pan_offset[1] = center_y - (center_y - pan_offset[1]) * (zoom_scale / old_zoom)
            
        elif event == cv2.EVENT_MBUTTONUP or event == cv2.EVENT_LBUTTONUP:
            panning = False
            last_pan_pos = None

    def on_key_press(event):
        nonlocal current_box_idx, current_sequence, all_sequences, running
        nonlocal zoom_scale, pan_offset, annotations, current_sequence_idx
        nonlocal show_help, entering_symbol, symbol_input, current_symbol

        scan_code_mapping = {
            49: 'n',     # n/т
            44: 'z',     # z/я
            32: 'd',     # d/в
            30: 'a',     # a/ф
            31: 's',     # s/ы
            35: 'h',     # h/р
            19: 'r',     # r/к
            18: 'e',     # e/у
            16: 'q',     # q/й
            45: 'x',     # x/ч
            20: 't',     # t/е
        }
        
        # If entering a symbol, handle text input
        if entering_symbol:
            if event.scan_code == 1:  # ESC key
                entering_symbol = False
                symbol_input = ""
                return
            elif event.scan_code == 28:  # Enter key
                current_symbol = symbol_input
                annotations[current_box_idx] = (annotations[current_box_idx][0], annotations[current_box_idx][1], current_symbol)
                entering_symbol = False
                symbol_input = ""
                print(f"Symbol set to: '{current_symbol}'")
                return
            elif event.scan_code == 14:  # Backspace key
                symbol_input = symbol_input[:-1] if symbol_input else ""
                return
            elif event.name and len(event.name) == 1:  # Regular character
                symbol_input += event.name
                return
            return
        
        actual_key = scan_code_mapping.get(event.scan_code, event.name)
        
        if actual_key == 't':  # Enter symbol mode
            entering_symbol = True
            symbol_input = current_symbol
            return
        
        elif actual_key == 'n':  # New sequence
            if current_sequence:  # Save current sequence if not empty
                all_sequences.append(current_sequence[:])
            current_sequence = []  # Start new sequence
            current_sequence_idx = len(all_sequences)
            print("Started new sequence")
            
        elif actual_key == 'e':  # Next sequence
            if len(all_sequences) > 0:
                # Save current sequence if it exists and we were editing an existing sequence
                if current_sequence and 0 <= current_sequence_idx < len(all_sequences):
                    all_sequences[current_sequence_idx] = current_sequence[:]
                
                # Move to next sequence
                if current_sequence_idx < len(all_sequences) - 1:
                    current_sequence_idx += 1
                else:
                    current_sequence_idx = 0
                current_sequence = all_sequences[current_sequence_idx][:]
                print(f"Editing sequence {current_sequence_idx}")
                
        elif actual_key == 'q':  # Previous sequence
            if len(all_sequences) > 0:
                # Save current sequence if it exists and we were editing an existing sequence
                if current_sequence and 0 <= current_sequence_idx < len(all_sequences):
                    all_sequences[current_sequence_idx] = current_sequence[:]
                
                # Move to previous sequence
                if current_sequence_idx > 0:
                    current_sequence_idx -= 1
                else:
                    current_sequence_idx = len(all_sequences) - 1
                current_sequence = all_sequences[current_sequence_idx][:]
                print(f"Editing sequence {current_sequence_idx}")
                
        elif actual_key == 'z':
            if current_sequence:
                current_sequence.pop()
                # Если редактируем существующую последовательность, обновляем её
                if current_sequence_idx >= 0 and current_sequence_idx < len(all_sequences):
                    all_sequences[current_sequence_idx] = current_sequence[:]
                print("Last point removed")
            
        elif actual_key == 'delete':  # Удаление текущей последовательности
            if current_sequence_idx >= 0 and current_sequence_idx < len(all_sequences):
                all_sequences.pop(current_sequence_idx)
                if len(all_sequences) > 0:
                    current_sequence_idx = min(current_sequence_idx, len(all_sequences) - 1)
                    current_sequence = all_sequences[current_sequence_idx][:]
                else:
                    current_sequence = []
                    current_sequence_idx = -1
                print(f"Sequence {current_sequence_idx} deleted")
        
        elif actual_key == 'd':
            # Save current sequence and move to next box
            if current_sequence:
                all_sequences.append(current_sequence[:])
            current_box_idx = min(current_box_idx + 1, len(annotations) - 1)
            current_sequence.clear()
            current_sequence_idx = -1
            all_sequences = annotations[current_box_idx][1]
            current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
            zoom_scale = 1.0
            pan_offset = [0, 0]
            print(f"Moved to next box: {current_box_idx}")
        
        elif actual_key == 'a':
            # Save current sequence and move to previous box
            if current_sequence:
                all_sequences.append(current_sequence[:])
            current_box_idx = max(current_box_idx - 1, 0)
            current_sequence.clear()
            current_sequence_idx = -1
            all_sequences = annotations[current_box_idx][1]
            current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
            zoom_scale = 1.0
            pan_offset = [0, 0]
            print(f"Moved to previous box: {current_box_idx}")
        
        elif actual_key == 'h':
            show_help = not show_help
            print("Help toggled")
        
        elif actual_key == 's' and output_path:
            if current_sequence:
                all_sequences.append(current_sequence[:])
            save_annotations(annotations, output_path)
            print(f"Annotations saved to: {output_path}")
        
        elif actual_key == 'x':  # Delete current bbox
            if len(annotations) > 0:
                # Remove current bbox
                annotations.pop(current_box_idx)
                
                if not annotations:  # If no boxes left
                    running = False
                    print("No boxes left to annotate")
                    return
                
                # Adjust current_box_idx if needed
                if current_box_idx >= len(annotations):
                    current_box_idx = len(annotations) - 1
                
                # Reset current sequence and update all_sequences for new current box
                current_sequence = []
                current_sequence_idx = -1
                all_sequences = annotations[current_box_idx][1]
                current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
                zoom_scale = 1.0
                pan_offset = [0, 0]
                print(f"Bbox deleted. Moved to box: {current_box_idx}")

    # Setup window and callbacks
    window_name = "Keypoint Annotation"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    # Register keyboard handlers for specific scan codes
    keys_to_monitor = [
        'n', 'z', 'd', 'a', 's', 'h', 'r', 'x', 'delete', 'backspace', 'e', 'q', 't'  # Added 't' for text entry
    ]
    
    for key in keys_to_monitor:
        keyboard.on_press_key(key, on_key_press, suppress=True)

    while running and current_box_idx < len(annotations):
        box = annotations[current_box_idx][0]
        
        # Convert to float first, then to int for display
        x = int(float(box[0]))
        y = int(float(box[1]))
        w = int(float(box[2]))
        h = int(float(box[3]))
        
        
        # Extract and resize box region
        box_img = image[y:y+h, x:x+w].copy()
        box_img = cv2.resize(box_img, (display_width, display_height))
        
        # Apply zoom and pan
        M = np.float32([[zoom_scale, 0, pan_offset[0]], 
                       [0, zoom_scale, pan_offset[1]]])
        display_img = cv2.warpAffine(box_img, M, (display_width, display_height))
        
        # Draw existing sequences
        for sequence in all_sequences:
            for i, (px, py) in enumerate(sequence):
                # Transform point coordinates
                screen_x = int(px * zoom_scale + pan_offset[0])
                screen_y = int(py * zoom_scale + pan_offset[1])
                
                cv2.circle(display_img, (screen_x, screen_y), 3, (0, 255, 0), -1)
                if i > 0:
                    prev_x = int(sequence[i-1][0] * zoom_scale + pan_offset[0])
                    prev_y = int(sequence[i-1][1] * zoom_scale + pan_offset[1])
                    cv2.line(display_img, (prev_x, prev_y), (screen_x, screen_y), (0, 255, 0), 1)
        
        # Draw current sequence
        for i, (px, py) in enumerate(current_sequence):
            screen_x = int(px * zoom_scale + pan_offset[0])
            screen_y = int(py * zoom_scale + pan_offset[1])
            
            cv2.circle(display_img, (screen_x, screen_y), 3, (0, 0, 255), -1)
            # Only draw line if there's a previous point
            if i > 0 and i - 1 < len(current_sequence):
                prev_x = int(current_sequence[i-1][0] * zoom_scale + pan_offset[0])
                prev_y = int(current_sequence[i-1][1] * zoom_scale + pan_offset[1])
                cv2.line(display_img, (prev_x, prev_y), (screen_x, screen_y), (0, 0, 255), 1)
        
        # Display current symbol or prompt
        if entering_symbol:
            prompt_text = f"Enter symbol: {symbol_input}_"
            cv2.rectangle(display_img, (10, 10), (300, 40), (0, 0, 0), -1)
            cv2.putText(display_img, prompt_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            if current_symbol:
                symbol_text = f"Symbol: '{current_symbol}'"
                cv2.rectangle(display_img, (10, 10), (200, 40), (0, 0, 0), -1)
                cv2.putText(display_img, symbol_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw help overlay
        display_img = draw_help(display_img)
        
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) == 27:  # ESC key
            running = False

    # Cleanup
    keyboard.unhook_all()
    cv2.destroyAllWindows()
    
    return annotations