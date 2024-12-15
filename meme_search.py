import os
import json
import torch
from PIL import Image
from art import text2art
from transformers import CLIPProcessor, CLIPModel
import numpy as np


class MemeSearchEngine:
    def __init__(
        self, image_dir="meme_images", model_name="openai/clip-vit-base-patch32"
    ):
        self.image_dir = image_dir
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.meme_urls = self.load_meme_urls()
        self.images, self.image_names = self.load_images()
        self.image_features = self.encode_images()

    def generate_json_from_folder(
        self, folder_path, output_file="meme_urls.json", base_url="meme_images/"
    ):
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} does not exist.")
            return

        images = [
            img
            for img in os.listdir(folder_path)
            if img.lower().endswith(("jpg", "jpeg", "png", "gif", "webp"))
        ]
        meme_data = {img: f"{base_url}{img}" for img in images}

        with open(output_file, "w") as f:
            json.dump(meme_data, f, indent=4)

        print(f"JSON file '{output_file}' created with {len(meme_data)} entries.")

    def load_meme_urls(self, json_file="meme_urls.json"):
        with open(json_file, "r") as f:
            return json.load(f)

    def load_images(self):
        images = []
        image_names = []
        for img_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                image_names.append(img_name)
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
        return images, image_names

    def encode_images(self):
        image_inputs = self.processor(
            images=self.images, return_tensors="pt", padding=True
        )["pixel_values"]
        with torch.no_grad():
            image_features = self.model.get_image_features(image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def search_memes(self, query):
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        scores = (text_features @ self.image_features.T).squeeze(0)
        scores = scores.detach().numpy()

        top_indices = scores.argsort()[-5:][::-1]
        results = [
            (self.image_names[i], self.meme_urls[self.image_names[i]], scores[i])
            for i in top_indices
        ]

        return results


def main():
    meme_search_engine = MemeSearchEngine()
    text_art = text2art("Meme Search Engine")
    print(text_art)
    print("\nType 'exit' to quit.")
    while True:
        query = input("\nEnter your search text: ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        results = meme_search_engine.search_memes(query)
        print("\nTop 5 Matches:")
        for name, url, score in results:
            print(f"Image: {name}, URL: {url}, Score: {score:.4f}")


if __name__ == "__main__":
    main()
