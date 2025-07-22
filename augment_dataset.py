import os
import random
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms
from tqdm import tqdm

input_dir = "./data/originals"
output_dir = "./data/augmented"
aug_per_image = 20  # number of augmented images per original

os.makedirs(output_dir, exist_ok=True)


augmentation_transforms = transforms.Compose([
    transforms.Resize((50, 250)),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor()
])

# Reconvert tensor back to PIL for saving
to_pil = transforms.ToPILImage()

print("Augmenting dataset...\n")
for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    base_name = os.path.splitext(filename)[0]
    img_path = os.path.join(input_dir, filename)
    image = Image.open(img_path).convert("RGB")

    for i in range(aug_per_image):
        augmented = augmentation_transforms(image)
        augmented_img = to_pil(augmented)

        
        if random.random() < 0.3:
            augmented_img = augmented_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

        # Save with original name + suffix
        out_filename = f"{base_name}_{i}.jpg"
        augmented_img.save(os.path.join(output_dir, out_filename))

print(f"\n✅ Done! Augmented images saved to: {output_dir}")
