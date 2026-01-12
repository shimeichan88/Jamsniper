# --- VISUALIZER (UPDATED: Removed Filter) ---
def draw_interface(data, shift, tilt):
    img = data['image'].copy() 
    results = data['results']
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Calibration Logic
    base_top = width * 0.60
    base_bottom = width * 0.40
    top_x = base_top + (width * shift) + (width * tilt)
    bottom_x = base_bottom + (width * shift) - (width * tilt)
    draw.line([(top_x, 0), (bottom_x, height)], fill="yellow", width=5)
    
    to_johor = 0
    to_woodlands = 0
    slope = (bottom_x - top_x) / height
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        box_w = x2 - x1
        box_h = y2 - y1
        
        # ⚠️ REMOVED THE "FLAT SHAPE" FILTER HERE
        # We still keep the "Billboard" filter (top left corner)
        if center_y > (height * 0.60) and center_x < (width * 0.30): continue 

        # Count Logic
        divider_x = top_x + (slope * center_y)
        if center_x < divider_x:
            to_johor += 1
            color = "#00ff00"
        else:
            to_woodlands += 1
            color = "#ff0000"
            
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
    return img, to_johor, to_woodlands