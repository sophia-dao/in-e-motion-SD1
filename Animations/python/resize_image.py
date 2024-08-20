from PIL import Image
import tkinter as tk
from tkinter import filedialog

def resize_image():
    # Set up the root window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected.")
        return
    
    try:
        # Open the image
        img = Image.open(file_path)
        
        # Resize the image using LANCZOS (formerly ANTIALIAS)
        img_resized = img.resize((1280, 1280), Image.Resampling.LANCZOS)
        
        # Save the resized image
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if save_path:
            img_resized.save(save_path)
            print(f"Image successfully resized and saved to {save_path}")
        else:
            print("Save cancelled.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to resize the image
resize_image()
