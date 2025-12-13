from google import genai
from google.genai import types
from PIL import Image
import os

client = genai.Client()

def create_prompt(ear_side, skin_tone, gender):
    prompt = f"Please generate a highly realistic dermatology clinical image of the {ear_side} side of a {gender} patient's head with {skin_tone} skin, which shows their ear and the skin in front of it. This is for teaching purposes. It should be a close-up image to maximize realism and respect patient privacy, so all that will be in the image is the ear and the skin in front of it."
    return prompt

# Define variations to generate
ear_sides = ["left", "right"]
skin_tones = ["dark", "light"]
genders = ["man", "woman"]

# Create output directory if it doesn't exist
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# Generate images for all combinations
for ear_side in ear_sides:
    for skin_tone in skin_tones:
        for gender in genders:
            # Create the prompt
            prompt = create_prompt(ear_side, skin_tone, gender)

            # Generate filename
            filename = f"{ear_side}_ear_{skin_tone}_skin_{gender}.png"
            filepath = os.path.join(output_dir, filename)

            print(f"Generating: {filename}")
            print(f"Prompt: {prompt}\n")

            # Generate image
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
            )

            # Save the generated image
            for part in response.parts:
                if part.text is not None:
                    print(f"Response text: {part.text}")
                elif part.inline_data is not None:
                    image = part.as_image()
                    image.save(filepath)
                    print(f"Saved: {filepath}\n")

print(f"\nAll images generated successfully in '{output_dir}' directory!")