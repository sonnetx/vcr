import os
import time
from pathlib import Path
from PIL import Image
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# --- CONFIGURATION ---
INPUT_DIR = "/scratch/users/sonnet/ddi"
OUTPUT_BASE_DIR = "/home/groups/roxanad/sonnet/vcr/processed_images"
RULER_DIR = Path(OUTPUT_BASE_DIR) / "with_ruler"
BLUE_INK_DIR = Path(OUTPUT_BASE_DIR) / "with_blue_ink"
NEITHER_DIR = Path(OUTPUT_BASE_DIR) / "neither"
BORDER_SIZE = 50 # Pixels

# --- GEMINI SETUP ---
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure the GEMINI_API_KEY environment variable is set correctly.")
    exit()

# System instruction to guide the model's classification
CLASSIFICATION_INSTRUCTION = """
You are an image classifier. Your task is to analyze the provided image and classify it based on the presence of a 'ruler' or 'blue ink' artifacts.

Output ONLY ONE of the following three keywords:
1. 'RULER' (if a ruler or measuring device is clearly visible)
2. 'BLUE_INK' (if there are scattered blue dots or smudges, indicating a blue ink artifact)
3. 'NEITHER' (if neither a ruler nor blue ink artifacts are present)

DO NOT include any other text, explanation, or punctuation.
"""

# --- EXPONENTIAL BACKOFF AND RETRY CONFIGURATION ---
# The @retry decorator will wrap the API call logic.
# 
# wait_random_exponential: Waits 2^x seconds, plus a random jitter, increasing exponentially.
#                         min=1: Wait at least 1 second.
#                         max=60: Maximum wait time is 60 seconds.
# stop_after_attempt(5): Stop retrying after 5 attempts total (1 original + 4 retries).
# retry_if_exception_type(APIError): Only retry if the error is a Google GenAI APIError, 
#                                    which includes rate limit (429) and transient server errors (500/503).
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(APIError),
    reraise=True # Re-raise the exception if max attempts are reached
)
def classify_image_with_gemini(image_path: Path) -> str:
    """
    Passes an image to Gemini-2.5-flash for classification with retries.
    Returns the classification keyword (RULER, BLUE_INK, or NEITHER).
    """
    print(f"  -> Classifying {image_path.name}...")
    
    # 1. Open the image file
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"  **ERROR**: Failed to open image {image_path.name} before API call: {e}")
        return 'NEITHER'

    # 2. Call the Gemini API with the image and system instruction
    # The @retry decorator handles the API call and any subsequent retries.
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[img, "Classify this image."],
        config=genai.types.GenerateContentConfig(
            system_instruction=CLASSIFICATION_INSTRUCTION,
            temperature=0.0
        )
    )
    
    # 3. Process and return the classification
    classification = response.text.strip().upper()
    
    if classification not in ['RULER', 'BLUE_INK', 'NEITHER']:
        print(f"  **WARNING**: Unexpected classification '{classification}'. Falling back to NEITHER.")
        return 'NEITHER'
            
    return classification


def add_white_border(image_path: Path, output_path: Path, border_size: int):
    """
    Loads an image, adds a white border, and saves it to the output path.
    (This function does not need backoff as it's a local file operation).
    """
    try:
        img = Image.open(image_path)
        new_width = img.width + 2 * border_size
        new_height = img.height + 2 * border_size
        bordered_img = Image.new(img.mode, (new_width, new_height), 'white')
        bordered_img.paste(img, (border_size, border_size))
        bordered_img.save(output_path)
        print(f"  -> Bordered and saved to {output_path.name}")
    except Exception as e:
        print(f"  **ERROR**: Failed to add border to {image_path.name}: {e}")


def main():
    """
    Main function to orchestrate the image classification, bordering, and sorting.
    """
    print(f"🚀 Starting Image Processor...")
    print(f"🔍 Scanning images in: {INPUT_DIR}")
    
    # 1. Create output directories
    RULER_DIR.mkdir(parents=True, exist_ok=True)
    BLUE_INK_DIR.mkdir(parents=True, exist_ok=True)
    NEITHER_DIR.mkdir(parents=True, exist_ok=True)
    
    image_files = [
        p for p in Path(INPUT_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']
    ]
    
    if not image_files:
        print(f"\n⚠️ No images found in the '{INPUT_DIR}' directory. Check the path and file extensions.")
        return

    total_count = len(image_files)
    print(f"✅ Found {total_count} images to process.")
    
    # 2. Process each image
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing Image {i+1}/{total_count}: {image_path.name}")
        
        classification = 'NEITHER' # Default/Fallback
        
        # --- Classification Step with Retries ---
        try:
            classification = classify_image_with_gemini(image_path)
        except APIError as e:
            # This block is reached if all 5 retry attempts failed
            print(f"  🛑 **CRITICAL ERROR**: Final retry failed for {image_path.name}. APIError: {e}")
            print("  -> Image will be placed in the 'neither' directory.")
        except Exception as e:
            # Handle other unexpected errors not covered by APIError
            print(f"  🛑 **UNEXPECTED ERROR**: Failed to classify {image_path.name}: {e}")
            print("  -> Image will be placed in the 'neither' directory.")
        
        print(f"  -> **FINAL CLASSIFICATION**: {classification}")
        
        # --- Sorting and Bordering Step ---
        if classification == 'RULER':
            target_dir = RULER_DIR
        elif classification == 'BLUE_INK':
            target_dir = BLUE_INK_DIR
        else: # Covers 'NEITHER' and all error/fallback cases
            target_dir = NEITHER_DIR
            
        output_path = target_dir / image_path.name
        add_white_border(image_path, output_path, BORDER_SIZE)
        
    print("\n🎉 **Processing Complete!**")
    print(f"Results are sorted into the following directories under {OUTPUT_BASE_DIR}.")


if __name__ == "__main__":
    main()