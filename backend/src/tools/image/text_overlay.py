from langchain_core.tools import tool
from typing import Dict, Optional
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


@tool
def text_overlay_tool(
    image_path: str,
    vision_response: str,
    text_replacements: Dict[str, str],
    font_path: Optional[str] = None,
    background_fill: bool = True,
    output_path: Optional[str] = None
) -> str:
    """
    Replace text in images with new text while maintaining similar font size and positioning.
    
    This tool uses Google Cloud Vision API response to locate text and replace it with
    new text while automatically calculating appropriate font size and positioning.
    
    Args:
        image_path: Path to the input image file
        vision_response: JSON response from detect_text_tool containing text annotations
        text_replacements: Dictionary mapping original text to replacement text
                          e.g. {"HELLO": "HOLA", "WORLD": "MUNDO"}
        font_path: Path to TTF font file (uses default system font if None)
        background_fill: Whether to fill the original text area with background color
        output_path: Path for output image (auto-generated if None)
    
    Returns:
        Path to the output image with replaced text
    """
    try:
        # Validate input file
        if not os.path.exists(image_path):
            return f"❌ Error: Image file not found: {image_path}"
        
        # Parse vision response
        try:
            vision_data = json.loads(vision_response)
        except json.JSONDecodeError:
            return "❌ Error: Invalid JSON in vision_response"
        
        # Load image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        draw = ImageDraw.Draw(image)
        
        # Generate output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_text_replaced{input_path.suffix}")
        
        print(f"🔄 Replacing text in image: {image_path}")
        
        # Get text annotations from vision response
        text_annotations = []
        if 'textAnnotations' in vision_data:
            text_annotations = vision_data['textAnnotations']
        elif 'responses' in vision_data and vision_data['responses']:
            if 'textAnnotations' in vision_data['responses'][0]:
                text_annotations = vision_data['responses'][0]['textAnnotations']
        
        if not text_annotations:
            return "❌ Error: No text annotations found in vision response"
        
        # Skip the first annotation (it's the full text) and process individual words/phrases
        replacements_made = 0
        
        for annotation in text_annotations[1:]:  # Skip first annotation (full text)
            original_text = annotation.get('description', '').strip()
            
            # Check if this text should be replaced
            replacement_text = None
            for orig, repl in text_replacements.items():
                if orig.upper() == original_text.upper():
                    replacement_text = repl
                    break
            
            if replacement_text is None:
                continue
            
            # Get bounding box
            bounding_poly = annotation.get('boundingPoly', {})
            vertices = bounding_poly.get('vertices', [])
            
            if len(vertices) < 4:
                continue
            
            # Calculate bounding box dimensions
            x_coords = [v.get('x', 0) for v in vertices]
            y_coords = [v.get('y', 0) for v in vertices]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            if box_width <= 0 or box_height <= 0:
                continue
            
            # Calculate font size based on bounding box height
            font_size = max(12, int(box_height * 0.7))
            
            # Load font
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    # Try to load default system fonts
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Get text dimensions for the replacement text
            bbox = draw.textbbox((0, 0), replacement_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Adjust font size if text is too wide for the bounding box
            if text_width > box_width * 0.95:
                scale_factor = (box_width * 0.95) / text_width
                new_font_size = max(8, int(font_size * scale_factor))
                try:
                    if font_path and os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, new_font_size)
                    else:
                        try:
                            font = ImageFont.truetype("arial.ttf", new_font_size)
                        except:
                            font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                # Recalculate text dimensions
                bbox = draw.textbbox((0, 0), replacement_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            
            # Fill background if requested
            if background_fill:
                # Sample background color from the area around the text
                background_color = _sample_background_color(image, x_min, y_min, x_max, y_max)
                draw.rectangle([x_min, y_min, x_max, y_max], fill=background_color)
            
            # Calculate text position (center in bounding box)
            text_x = x_min + (box_width - text_width) // 2
            text_y = y_min + (box_height - text_height) // 2
            
            # Determine text color
            text_color = _determine_text_color(image, x_min, y_min, x_max, y_max)
            
            # Draw the replacement text
            draw.text((text_x, text_y), replacement_text, font=font, fill=text_color)
            
            replacements_made += 1
            print(f"✅ Replaced '{original_text}' with '{replacement_text}'")
        
        # Save the modified image
        image.save(output_path, quality=95)
        
        if replacements_made == 0:
            return f"⚠️ No text replacements made. Output saved to: {output_path}"
        
        return f"✅ Successfully replaced {replacements_made} text(s)! Output saved to: {output_path}"
        
    except Exception as e:
        return f"❌ Error during text overlay: {str(e)}"


def _sample_background_color(image, x_min, y_min, x_max, y_max):
    """
    Sample the background color around the text area.
    
    Args:
        image: PIL Image object
        x_min, y_min, x_max, y_max: Text bounding box coordinates
        
    Returns:
        Background color as RGB tuple
    """
    try:
        # Expand the sampling area slightly beyond the text box
        margin = 5
        sample_x_min = max(0, x_min - margin)
        sample_y_min = max(0, y_min - margin)
        sample_x_max = min(image.width, x_max + margin)
        sample_y_max = min(image.height, y_max + margin)
        
        # Sample pixels from the border area
        border_pixels = []
        
        # Top and bottom borders
        for x in range(sample_x_min, sample_x_max):
            if sample_y_min < image.height:
                border_pixels.append(image.getpixel((x, sample_y_min)))
            if sample_y_max - 1 < image.height and sample_y_max - 1 >= 0:
                border_pixels.append(image.getpixel((x, sample_y_max - 1)))
        
        # Left and right borders
        for y in range(sample_y_min, sample_y_max):
            if sample_x_min < image.width:
                border_pixels.append(image.getpixel((sample_x_min, y)))
            if sample_x_max - 1 < image.width and sample_x_max - 1 >= 0:
                border_pixels.append(image.getpixel((sample_x_max - 1, y)))
        
        if border_pixels:
            # Calculate average color
            avg_r = sum(p[0] for p in border_pixels) // len(border_pixels)
            avg_g = sum(p[1] for p in border_pixels) // len(border_pixels)
            avg_b = sum(p[2] for p in border_pixels) // len(border_pixels)
            return (avg_r, avg_g, avg_b)
        
    except Exception:
        pass
    
    # Default to white if sampling fails
    return (255, 255, 255)


def _determine_text_color(image, x_min, y_min, x_max, y_max):
    """
    Determine appropriate text color based on background brightness.
    
    Args:
        image: PIL Image object
        x_min, y_min, x_max, y_max: Text bounding box coordinates
        
    Returns:
        Text color as RGB tuple (black or white)
    """
    try:
        # Sample the background color
        bg_color = _sample_background_color(image, x_min, y_min, x_max, y_max)
        
        # Calculate brightness (perceived luminance)
        brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        
        # Use black text on light backgrounds, white text on dark backgrounds
        if brightness > 127:
            return (0, 0, 0)  # Black
        else:
            return (255, 255, 255)  # White
            
    except Exception:
        return (0, 0, 0)  # Default to black 