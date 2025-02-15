# Meme Search

A Meme Search Engine that allows users to search for memes by keywords. The search engine uses a pre-trained model from OpenAI to extract features from the memes and then uses these features to find the most similar memes to the query.

## How to run locally

1. Clone the repository
```bash
 git clone ssh://submission.fringecore.sh/5P8Gjb/fringecore-machine-learning-challenge-meme-search
```
2. Change the directory to the repository
```bash
cd fringecore-machine-learning-challenge-meme-search
```
3. Set up a virtual environment using
```bash
python3 -m venv .venv
```
4. Activate the virtual environment using
```bash
source .venv/bin/activate
```
5. Create a .env file and Add the Gemini API key to the .env file by following the format in the .env.example file


6. Install the required packages inside virtual env using
```bash
pip3 install -r requirements.txt
```
7. Run the application using
```bash
python3 meme_search.py
```