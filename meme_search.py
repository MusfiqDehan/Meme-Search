import os
import json
from typing import List, Dict
from art import text2art
import google.generativeai as genai
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MemeSearchEngine:
    def __init__(self, meme_folder="meme_images"):
        """
        Initialize the Meme Search Engine
        """
        try:
            # Configure Gemini API
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                api_key = input("Please enter your GEMINI_API_KEY: ")
                if not api_key:
                    raise ValueError(
                        "No API key provided. Please set GEMINI_API_KEY in .env file or enter it at runtime."
                    )

            genai.configure(api_key=api_key)

            # Initialize vision and text models
            self.vision_model = genai.GenerativeModel("gemini-1.5-flash")
            self.text_model = genai.GenerativeModel("gemini-1.5-flash")

            # Set up meme image folder
            self.meme_folder = meme_folder
            self.meme_images = self._load_meme_images()

        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def _load_meme_images(self) -> List[Dict]:
        meme_data = []

        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

        for filename in os.listdir(self.meme_folder):
            # Check file extension
            if os.path.splitext(filename)[1].lower() not in image_extensions:
                continue

            full_path = os.path.join(self.meme_folder, filename)

            try:
                # Open image
                image = Image.open(full_path)

                # Extract text from image
                text_response = self.vision_model.generate_content(
                    ["Extract all readable text from this image", image]
                )
                extracted_text = (
                    text_response.text.strip() if hasattr(text_response, "text") else ""
                )

                # Analyze image content
                content_response = self.vision_model.generate_content(
                    [
                        "Describe the contents, objects, and context of this image in detail",
                        image,
                    ]
                )
                image_description = (
                    content_response.text.strip()
                    if hasattr(content_response, "text")
                    else ""
                )

                # Combine extracted information
                meme_data.append(
                    {
                        "filename": filename,
                        "path": full_path,
                        "extracted_text": extracted_text,
                        "image_description": image_description,
                        "combined_text": f"{extracted_text} {image_description}",
                    }
                )

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return meme_data

    def search_meme_images(
        self, search_query: str, top_k: int = 5
    ) -> List[Dict[str, str]]:
        # Prepare search
        try:
            context_response = self.text_model.generate_content(
                f"Expand this search query to include related concepts and context: {search_query}"
            )
            expanded_query = (
                context_response.text.strip()
                if hasattr(context_response, "text")
                else search_query
            )
        except Exception:
            expanded_query = search_query

        # Extract texts from meme images
        meme_texts = [meme["combined_text"] for meme in self.meme_images]

        # Add expanded query to corpus
        corpus = meme_texts + [expanded_query]

        # Create TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        # Create results with multiple scoring mechanisms
        results = []
        for i, text_score in enumerate(similarity_scores):
            try:
                # Additional semantic similarity check
                semantic_response = self.text_model.generate_content(
                    f"Compare the semantic similarity between these two texts. "
                    f"Query: {expanded_query}\n"
                    f"Meme Text: {self.meme_images[i]['combined_text']}"
                )
                semantic_score = (
                    float(semantic_response.text.strip().split("%")[0]) / 100
                    if "%" in semantic_response.text
                    else 0
                )
            except Exception:
                semantic_score = 0

            # Combine scoring methods
            combined_score = (text_score * 0.6 + semantic_score * 0.4) * 100

            results.append(
                {
                    "filename": self.meme_images[i]["filename"],
                    "path": self.meme_images[i]["path"],
                    "extracted_text": self.meme_images[i]["extracted_text"],
                    "image_description": self.meme_images[i]["image_description"],
                    "text_score": text_score * 100,
                    "semantic_score": semantic_score * 100,
                    "combined_score": combined_score,
                }
            )

        # Sort results
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        return results[:top_k]

    def run_search_loop(self):
        """
        Interactive loop for searching meme images
        """
        print("\n🤖 Meme Image Search Engine 🤖")
        print(f"Loaded {len(self.meme_images)} meme images")
        print("Enter 'exit' to quit the search")

        while True:
            # Get user input
            search_query = input("\nEnter meme search text: ").strip()

            # Check for exit condition
            if search_query.lower() == "exit":
                print("Exiting Meme Search Engine...")
                break

            # Validate input
            if not search_query:
                print("Please enter a valid search text.")
                continue

            # Perform search
            try:
                results = self.search_meme_images(search_query)

                # Display results
                if results:
                    print(f"\nTop {len(results)} Meme Images for '{search_query}':")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. Filename: {result['filename']}")
                        print(f"   Path: {result['path']}")
                        # print(f"   Extracted Text: {result['extracted_text']}")
                        # print(f"   Image Description: {result['image_description']}")
                        # print(f"   Text Similarity Score: {result['text_score']:.2f}%")
                        # print(
                        #     f"   Semantic Similarity Score: {result['semantic_score']:.2f}%"
                        # )
                        print(
                            f"   Combined Relevance Score: {result['combined_score']:.2f}%\n"
                        )
                else:
                    print("No meme images found for the search query.")

            except Exception as e:
                print(f"An error occurred during search: {e}")


def main():
    try:
        meme_search = MemeSearchEngine()
        meme_search.run_search_loop()

    except Exception as e:
        print(f"Critical error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
