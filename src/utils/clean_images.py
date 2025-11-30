import os
from pathlib import Path
from PIL import Image
from google import genai
from google.genai.errors import APIError
from typing import Union
import io # To handle image data in memory

# --- CONFIGURATION ---
INPUT_BASE_DIR = "/home/groups/roxanad/sonnet/vcr/processed_images" # This is the output_base_dir from the previous script
RULER_INPUT_DIR = Path(INPUT_BASE_DIR) / "with_ruler"
BLUE_INK_INPUT_DIR = Path(INPUT_BASE_DIR) / "with_blue_ink"

CLEANED_OUTPUT_DIR = "/home/groups/roxanad/sonnet/vcr/cleaned_images"
CROPPED_BORDER_SIZE = 50 # Must match the border size added in the previous script

# --- GEMINI SETUP ---
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    exit()

# --- GEMINI INPAINTING FUNCTION ---
def remove_objects_with_gemini(image_path: Path, prompt: str) -> Union[Image.Image, None]:
    """
    Sends an image to Gemini for object removal (inpainting) based on the prompt.
    Returns the modified PIL Image object or None on failure.
    """
    print(f"  -> Sending {image_path.name} to Gemini for object removal...")
    print(f"     Prompt: '{prompt}'")

    try:
        # Load the image for the API call
        img_pil = Image.open(image_path)
        
        new_width = img_pil.width + 2 * CROPPED_BORDER_SIZE
        new_height = img_pil.height + 2 * CROPPED_BORDER_SIZE
        bordered_img = Image.new(img_pil.mode, (new_width, new_height), 'white')
        bordered_img.paste(img_pil, (CROPPED_BORDER_SIZE, CROPPED_BORDER_SIZE))

        model_name = 'gemini-2.5-flash-image' 

        response = client.models.generate_content(
            model=model_name,
            contents=[bordered_img, prompt]
        )
        
        # Check if the response contains an image
        if not response.parts:
            print(f"  **ERROR**: Gemini returned no content for {image_path.name}.")
            return None
        
        # Find the image part in the response
        generated_image_part = None
        for part in response.parts:
            if hasattr(part, 'mime_type') and part.mime_type.startswith('image/'):
                generated_image_part = part
                break
        
        if generated_image_part is None:
            print(f"  **ERROR**: Gemini did not return an image for {image_path.name}.")
            return None

        # Convert the byte data from the response into a PIL Image
        img_bytes = io.BytesIO(generated_image_part.data)
        modified_img = Image.open(img_bytes)
        print(f"  -> Gemini successfully processed {image_path.name}.")
        return modified_img

    except APIError as e:
        print(f"  **ERROR**: Gemini API call failed for {image_path.name}: {e}")
        return None
    except Exception as e:
        print(f"  **ERROR**: Failed to open, send, or receive image {image_path.name}: {e}")
        return None

def crop_border(image: Image.Image, border_size: int) -> Image.Image:
    """
    Crops a specified border size from all sides of a PIL Image.
    """
    width, height = image.size
    left = border_size
    top = border_size
    right = width - border_size
    bottom = height - border_size
    
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def main():
    """
    Main function to orchestrate object removal and border cropping.
    """
    print(f"🚀 Starting Image Cleaner and Cropper...")
    
    # 1. Create output directory if it doesn't exist
    Path(CLEANED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    processed_count = 0

    # --- Process images from the RULER directory ---
    print(f"\nProcessing images from: {RULER_INPUT_DIR}")
    ruler_images = list(RULER_INPUT_DIR.iterdir())
    
    if not ruler_images:
        print(f"No images found in '{RULER_INPUT_DIR}'.")
    
    for i, image_path in enumerate(ruler_images):
        if not image_path.is_file():
            continue
        print(f"\nImage {i+1}/{len(ruler_images)} (Ruler): {image_path.name}")
        
        # Step 1: Remove ruler using Gemini
        modified_img = remove_objects_with_gemini(image_path, "Remove the ruler or measuring scale from the bottom of the image.")
        
        if modified_img:
            # Step 2: Crop the 50-pixel border
            cropped_img = crop_border(modified_img, CROPPED_BORDER_SIZE)
            
            # Step 3: Save the cleaned and cropped image
            output_path = Path(CLEANED_OUTPUT_DIR) / image_path.name
            cropped_img.save(output_path)
            print(f"  -> Cleaned, cropped, and saved: {output_path.name}")
            processed_count += 1
        else:
            print(f"  -> Skipping {image_path.name} due to prior error.")

    # --- Process images from the BLUE_INK directory ---
    print(f"\nProcessing images from: {BLUE_INK_INPUT_DIR}")
    blue_ink_images = list(BLUE_INK_INPUT_DIR.iterdir())

    if not blue_ink_images:
        print(f"No images found in '{BLUE_INK_INPUT_DIR}'.")
        
    for i, image_path in enumerate(blue_ink_images):
        if not image_path.is_file():
            continue
        print(f"\nImage {i+1}/{len(blue_ink_images)} (Blue Ink): {image_path.name}")
        
        # Step 1: Remove blue ink using Gemini
        modified_img = remove_objects_with_gemini(image_path, "Remove all blue ink marks, dots, or smudges from the image. Make sure that you are keeping the lesion and skin condition intact.")
        
        if modified_img:
            # Step 2: Crop the 50-pixel border
            cropped_img = crop_border(modified_img, CROPPED_BORDER_SIZE)
            
            # Step 3: Save the cleaned and cropped image
            output_path = Path(CLEANED_OUTPUT_DIR) / image_path.name
            cropped_img.save(output_path)
            print(f"  -> Cleaned, cropped, and saved: {output_path.name}")
            processed_count += 1
        else:
            print(f"  -> Skipping {image_path.name} due to prior error.")

    print(f"\n🎉 **Cleaning and Cropping Complete!** Processed {processed_count} images.")
    print(f"Cleaned images are saved in: {CLEANED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()