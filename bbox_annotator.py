import cv2
import numpy as np
from data_annotator.help_overlay import HelpOverlay
from data_annotator.file_processor import save_annotations
import tkinter as tk
from tkinter import ttk


class SymbolInputDialog:
    def __init__(self, current_symbol=""):
        self.result = None
        
        # Создаем главное окно
        self.root = tk.Tk()
        self.root.title("Ввод символа")
        
        # Устанавливаем стиль
        style = ttk.Style()
        style.configure('Custom.TFrame', background='#f0f0f0')
        style.configure('Custom.TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Custom.TButton', font=('Arial', 10), padding=5)
        style.configure('Custom.TEntry', font=('Arial', 10))
        
        # Устанавливаем положение окна по центру экрана
        window_width = 300
        window_height = 150
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Настраиваем внешний вид окна
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(False, False)
        
        # Создаем и размещаем элементы интерфейса
        main_frame = ttk.Frame(self.root, padding="20", style='Custom.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Введите символ:", style='Custom.TLabel').grid(row=0, column=0, columnspan=2, pady=(0,10))
        
        self.entry = ttk.Entry(main_frame, width=40, style='Custom.TEntry')
        self.entry.grid(row=1, column=0, columnspan=2, pady=(0,20))
        self.entry.insert(0, current_symbol)
        self.entry.focus()
        
        button_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        button_frame.grid(row=2, column=0, columnspan=2)
        
        ok_button = ttk.Button(button_frame, text="OK", command=self.ok_clicked, style='Custom.TButton', width=10)
        ok_button.grid(row=0, column=0, padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Отмена", command=self.cancel_clicked, style='Custom.TButton', width=10)
        cancel_button.grid(row=0, column=1, padx=5)
        
        # Привязываем клавиши
        self.root.bind('<Return>', lambda e: self.ok_clicked())
        self.root.bind('<Escape>', lambda e: self.cancel_clicked())
        self.root.protocol("WM_DELETE_WINDOW", self.cancel_clicked)
        
        # Делаем окно модальным
        self.root.transient()
        self.root.grab_set()
        
    def ok_clicked(self):
        self.result = self.entry.get()
        self.root.destroy()
        
    def cancel_clicked(self):
        self.root.destroy()
        
    def show(self):
        self.root.mainloop()
        return self.result


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
        
        # Transform coordinates back to original image space
        orig_x = (x - pan_offset[0]) / zoom_scale
        orig_y = (y - pan_offset[1]) / zoom_scale
        
        if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_SHIFTKEY):
            current_sequence.append((int(orig_x), int(orig_y)))
        
        elif event == cv2.EVENT_MBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY)):
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

    # Setup window and callbacks
    window_name = "Keypoint Annotation"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

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
        
        # Display current symbol
        if current_symbol:
            symbol_text = f"Symbol: '{current_symbol}'"
            # Создаем фон с прозрачностью
            overlay = display_img.copy()
            cv2.rectangle(overlay, (10, 10), (max(200, len(symbol_text) * 12), 40), (0, 0, 0), -1)
            # Применяем прозрачность
            alpha = 0.7
            display_img = cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0)
            # Добавляем текст
            cv2.putText(display_img, symbol_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Draw help overlay
        display_img = draw_help(display_img)
        
        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            running = False
        elif key == ord('t'):  # Enter symbol mode
            dialog = SymbolInputDialog(current_symbol)
            result = dialog.show()
            if result is not None:
                current_symbol = result
                annotations[current_box_idx] = (annotations[current_box_idx][0], annotations[current_box_idx][1], current_symbol)
                print(f"Symbol set to: '{current_symbol}'")
        elif key == ord('n'):  # New sequence
            if current_sequence:
                all_sequences.append(current_sequence[:])
            current_sequence = []
            current_sequence_idx = len(all_sequences)
            print("Started new sequence")
        elif key == ord('e'):  # Next sequence
            if len(all_sequences) > 0:
                if current_sequence and 0 <= current_sequence_idx < len(all_sequences):
                    all_sequences[current_sequence_idx] = current_sequence[:]
                if current_sequence_idx < len(all_sequences) - 1:
                    current_sequence_idx += 1
                else:
                    current_sequence_idx = 0
                current_sequence = all_sequences[current_sequence_idx][:]
                print(f"Editing sequence {current_sequence_idx}")
        elif key == ord('q'):  # Previous sequence
            if len(all_sequences) > 0:
                if current_sequence and 0 <= current_sequence_idx < len(all_sequences):
                    all_sequences[current_sequence_idx] = current_sequence[:]
                if current_sequence_idx > 0:
                    current_sequence_idx -= 1
                else:
                    current_sequence_idx = len(all_sequences) - 1
                current_sequence = all_sequences[current_sequence_idx][:]
                print(f"Editing sequence {current_sequence_idx}")
        elif key == ord('z'):  # Undo last keypoint
            if current_sequence:
                current_sequence.pop()
                if current_sequence_idx >= 0 and current_sequence_idx < len(all_sequences):
                    all_sequences[current_sequence_idx] = current_sequence[:]
                print("Last point removed")
        elif key == ord('d'):  # Next box
            if current_sequence:
                all_sequences.append(current_sequence[:])
            current_box_idx = min(current_box_idx + 1, len(annotations) - 1)
            current_sequence = []
            current_sequence_idx = -1
            all_sequences = annotations[current_box_idx][1]
            current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
            zoom_scale = 1.0
            pan_offset = [0, 0]
            print(f"Moved to next box: {current_box_idx}")
        elif key == ord('a'):  # Previous box
            if current_sequence:
                all_sequences.append(current_sequence[:])
            current_box_idx = max(current_box_idx - 1, 0)
            current_sequence = []
            current_sequence_idx = -1
            all_sequences = annotations[current_box_idx][1]
            current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
            zoom_scale = 1.0
            pan_offset = [0, 0]
            print(f"Moved to previous box: {current_box_idx}")
        elif key == ord('h'):  # Toggle help
            show_help = not show_help
            print("Help toggled")
        elif key == ord('s') and output_path:  # Save annotations
            if current_sequence:
                all_sequences.append(current_sequence[:])
            save_annotations(annotations, output_path)
            print(f"Annotations saved to: {output_path}")
        elif key == ord('x'):  # Delete current bbox
            if len(annotations) > 0:
                annotations.pop(current_box_idx)
                if not annotations:
                    running = False
                    print("No boxes left to annotate")
                    continue
                if current_box_idx >= len(annotations):
                    current_box_idx = len(annotations) - 1
                current_sequence = []
                current_sequence_idx = -1
                all_sequences = annotations[current_box_idx][1]
                current_symbol = annotations[current_box_idx][2] if len(annotations[current_box_idx]) > 2 else ""
                zoom_scale = 1.0
                pan_offset = [0, 0]
                print(f"Bbox deleted. Moved to box: {current_box_idx}")
        elif key == ord('r'):  # Reset view
            zoom_scale = 1.0
            pan_offset = [0, 0]
            print("View reset")

    cv2.destroyAllWindows()
    return annotations