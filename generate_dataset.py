import os
import random
import string
from PIL import Image
from stegano import lsb

def generate_random_payload(length=100):
    """Generates a random string of characters to act as the hidden message."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_lsb_dataset(cover_dir, output_dir, sample_size=2500):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_items = os.listdir(cover_dir)
    images = [f for f in all_items if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print(f"Error: No images found directly inside '{cover_dir}'.")
        return

    images = images[:sample_size]
    
    print(f"Generating {len(images)} LSB Stego images...")
    
    for idx, img_name in enumerate(images):
        cover_path = os.path.join(cover_dir, img_name)
        output_path = os.path.join(output_dir, img_name.rsplit('.', 1)[0] + '.png')
        
        # 1. Open the image and force it to RGB mode to prevent the [Y/n] prompt
        img = Image.open(cover_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 2. Generate a random secret message
        secret_message = generate_random_payload(length=150)
        
        # 3. Hide the message in the image (pass the Image object, not the path)
        secret_image = lsb.hide(img, secret_message)
        
        # 4. Save the infected image
        secret_image.save(output_path)
        
        # Print progress every 50 images
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(images)} images...")

    print("Class 1 (LSB) Dataset Generation Complete!")

# Run the generator
create_lsb_dataset(
    cover_dir='data/class_0_cover', 
    output_dir='data/class_1_lsb',
    sample_size=2500
)