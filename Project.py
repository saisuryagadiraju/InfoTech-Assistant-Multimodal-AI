import json
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import openai
from flask import Flask, request, jsonify, render_template, session
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os


app = Flask(__name__)
app.secret_key = "AIT526" # Required for session management as random name 

class InteractiveRAGChatBot:
    def __init__(self, data_path: str):
        # Load JSON data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Initialize SpaCy model for keyword extraction and sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize SentenceTransformer for encoding and set device to CPU
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True)

        self.device = "cpu"
        self.embedding_model.to(self.device)

        # Initialize T5 model and tokenizer for summarization
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

        # Pre-define greetings
        self.greetings = ["hi", "hello", "hey", "Hii", "hii", "how are you", "how about you", "quit", "exit", "bye"] #for user greetings  
        self.greeting_response = {
            "default": "Hello! How can I assist you today?",
            "how_are_you": "I'm good, focusing on InfoTechnology!, Thank you",
            "how about you": "I'm Happy and learning new NDE Technologies",
            "farewell": "Thank you so much! Hope you got the necessary information."
        }
        # Process and prepare data chunks from JSON
        self.chunks = self.process_data()

    def handle_greetings(self, user_input: str) -> str: #whhich handles the greetings logic
        """
        Handle user greetings and respond accordingly.
        """
        user_input_lower = user_input.lower()
        if "how are you" in user_input.lower():
            return self.greeting_response["how_are_you"]
        if "how about you" in user_input.lower():
            return self.greeting_response["how about you"]
        if any(farewell in user_input_lower for farewell in ["quit", "exit", "bye"]):
            return self.greeting_response["farewell"]
        for greeting in self.greetings:
            if greeting in user_input.lower():
                return self.greeting_response["default"]
        return None


    def process_data(self) -> List[Dict]:
        # Store relevant information from JSON into manageable chunks
        chunks = []
        for id_key, content in self.data.items():
            if "text" in content:
                chunks.append({
                    "text": content["text"], #for text 
                    "id": id_key,#postid location
                    "images": content.get("images", []) # for images
                })
        return chunks
    # Giving instruction to LLM model not to do any hallucinations  instead take the source content from source file itself
    def extract_keywords(self, user_input: str) -> List[str]:
        # Extract keywords from user input
        doc = self.nlp(user_input)
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        return keywords

    def search_data(self, keywords: List[str]) -> List[Dict]:
        # Find chunks containing any keyword to maximize relevant matches
        matched_chunks = []
        for chunk in self.chunks:
            if any(keyword.lower() in chunk["text"].lower() for keyword in keywords):
                matched_chunks.append(chunk)
        return matched_chunks

    def summarize_with_t5(self, text: str, max_length: int = 500) -> str: #summarizations
        """
        Use T5 for summarization. Dynamically adjust the summary length based on input size.
        """
        if len(text) <= max_length:
            return text  # No need to summarize if text is already short enough
        
        # Prepare input text for T5 summarization
        input_text = f"summarize: {text}"
        inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate summary with dynamic length adjustment
        summary_ids = self.t5_model.generate(
            inputs,
            max_length=max(min(150, len(text.split()) // 2), 50),  # Dynamically adjust the max length of summary
            min_length=30,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    #identify the content is from chuncks data or not
    def generate_response_from_chunks(self, user_input: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "I'm sorry, I couldn't find any relevant information for your query."

        # Select the best chunk based on keyword density and similarity
        query_embedding = self.embedding_model.encode([user_input], convert_to_numpy=True)
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)

        similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
        best_chunk = chunks[np.argmax(similarities)]

        # Use T5 summarization
        summarized_text = self.summarize_with_t5(best_chunk["text"])

        # Include images if available
        images = best_chunk.get("images", [])[:3]# We can also adjust the images based on the response generation
        if images:
            images_html = "<br>".join([f'<img src="{img}" alt="Related Image" width="300">' for img in images])
            summarized_text += f"<br><br>{images_html}"

        return summarized_text


    def generate_response(self, user_input: str) -> str:
        # Check for greetings
        greeting_response = self.handle_greetings(user_input)
        if greeting_response:
            return greeting_response

        # Extract keywords and search JSON data for relevant content
        keywords = self.extract_keywords(user_input)
        matched_chunks = self.search_data(keywords)
        chunk_response = ""
        json_context = ""

        # Define a similarity threshold and use JSON data if relevant
        similarity_threshold = 0.5 #we can adjust the limit as we like
        if matched_chunks:
            query_embedding = self.embedding_model.encode([user_input], convert_to_numpy=True)
            chunk_texts = [chunk["text"] for chunk in matched_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
            similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
            best_match_index = np.argmax(similarities)
            best_similarity_score = similarities[best_match_index]
            #to givee best output based on the similarity 
            if best_similarity_score >= similarity_threshold:
                best_chunk = matched_chunks[best_match_index]
                chunk_response = self.summarize_if_needed(best_chunk["text"], max_length=700)
                images = best_chunk.get("images", [])[:3]
                if images:
                    images_html = "<br>".join([f'<img src="{img}" alt="Related Image" width="300">' for img in images])
                    chunk_response += f"<br><br>{images_html}"
                json_context = chunk_response

        # Retrieve conversation history from session, initialize if not present
        conversation_history = session.get('conversation_history', [])
        
        # Appending the last 5 interactions from conversation history to create a session context
        session_context = "\n".join([f"User: {msg['content']}\nAssistant: {msg['content']}" 
                                     for msg in conversation_history[-5:]])
        prompt = f"{session_context}\n\nContext: {json_context}\nUser Question: {user_input}" if json_context else user_input

        # Generate a response from LLM using OpenAI-style API
        try:
            completion = openai.ChatCompletion.create(
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", #obtained from the LM studion through local API call
                messages=[{"role": "system", "content": "You are InfoTechnology Bridge Assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,# for the probability distribution 
            )
            llm_response = completion.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error connecting to LLM: {e}")
            llm_response = "I'm having trouble connecting to my language model."

        # Combine JSON and LLM responses
        combined_response = f"{chunk_response}<br><br><strong>LLM Response:</strong><br>{llm_response}" if chunk_response else llm_response

        # Update session history with the current interaction
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": combined_response})
        session['conversation_history'] = conversation_history  # Store updated history in session
        session.modified = True  # Ensure Flask saves the session data
        print(conversation_history)
        return combined_response #to create LLM response along with bot response (chunk content)

# Initialize the chatbot
chatbot = InteractiveRAGChatBot(data_path="C:/Users/samee/OneDrive/Desktop/InfoBridge_scraped_data/InfoBridge_scraped_data.json") # Need to specify the path correctly to load the files 

@app.route('/') #Flask app which enroute to the HTML file for input and output
def index():
    return render_template("index.html")

@app.route('/get_answer', methods=['POST']) 
def get_answer():
    user_input = request.json.get("question", "")#get the user input
    response = chatbot.generate_response(user_input)# post the answer
    return jsonify({"text": response})

@app.route('/scrape_data', methods=['POST'])
def scrape_data():
    return jsonify({"status": "Re-scraping initiated!"})#Rescraping initiation (Need best System Config that which can handle the process)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
