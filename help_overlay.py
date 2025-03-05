import cv2
import numpy as np

class HelpOverlay:
    """
    Class for displaying help information overlay in the bbox editor window
    """
    def __init__(self):
        self.help_text = [
            "Controls:",
            "Left Click & Drag - Move box",
            "Alt + Left Click & Drag - Create new box",
            "Drag handles - Resize box",
            "Arrow Keys - Move selected box by 1 pixel",
            "Left Click + Ctrl - Select multiple boxes",
            "Del/X - Delete selected box(es)",
            "Mouse Wheel - Zoom in/out",
            "Middle Mouse/Shift + Drag - Pan image",
            "S - Save boxes to file",
            "ESC - Finish editing",
            "H - Toggle help",
            "R - Reset view",
            "C - Clear all selections"
        ]
        self.is_visible = False
        
    def toggle(self):
        """Toggle help overlay visibility"""
        self.is_visible = not self.is_visible
        
    def draw(self, image):
        """
        Draw help overlay on the image if visible
        
        Args:
            image: Input image to draw overlay on
            
        Returns:
            image: Image with help overlay if visible
        """
        if not self.is_visible:
            return image
            
        overlay = image.copy()
        h, w = image.shape[:2]
        text_bg = np.zeros((h, w, 3), dtype=np.uint8)
        padding = 10
        start_x = 20
        start_y = 40
        
        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_height = 25
        max_width = max([cv2.getTextSize(text, font, font_scale, 1)[0][0] for text in self.help_text])
        bg_width = max_width + 2 * padding
        bg_height = len(self.help_text) * text_height + 2 * padding
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (start_x - padding, start_y - padding - text_height),
                     (start_x + bg_width, start_y + bg_height),
                     (0, 0, 0), -1)
        
        # Apply transparency
        alpha = 0.7
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Draw text
        for i, line in enumerate(self.help_text):
            y = start_y + i * text_height
            cv2.putText(image, line, (start_x, y), font, font_scale, (255, 255, 255), 1)
            
        return image
