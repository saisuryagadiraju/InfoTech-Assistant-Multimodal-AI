
# Infotech Chatbot System


The InfoBridge InfoTech Assistant is an NLP-driven chatbot system designed to dynamically respond to user queries by leveraging pre-scraped data, advanced language models, and Retrieval-Augmented Generation (RAG) techniques. This README provides a complete guide to the system requirements, installation, and usage.

System Requirements
To ensure smooth execution, the following system specifications are recommended: 
![image](https://github.com/user-attachments/assets/83d793d9-4684-41c1-bb67-25946a15359b)

Python Environment
Python Version: Python 3.11.5
Python IDEs Supported: VS Code, Jupyter Notebook, or any other local Python IDE
Virtual Environment: Required (Anaconda recommended)
Installation Guide
1. Setting Up the Virtual Environment
Install Anaconda if you haven’t already.
```
Open Anaconda Prompt and execute the following commands:
conda create -n ait526 python=3.11.5 -y
conda activate ait526
```
![image](https://github.com/user-attachments/assets/9422dcd8-160e-4203-a8a7-35f84eb062f2)

3. Installing Dependencies
Download the project files (ZIP) and extract them to a new folder on your system.
Open a terminal or command prompt in the project folder and install the required dependencies:
bash

Download
```
pip install -r requirements.txt
```
3. Installing LM Studio
Windows
Download LM Studio for Windows from this link
```
https://releases.lmstudio.ai/windows/0.2.31/candidate/b/LM-Studio-0.2.31-Setup.exe
```
macOS
Download LM Studio for macOS from this link.
```
For MacOS:
https://releases.lmstudio.ai/mac/arm64/0.2.31/2/LM-Studio-0.2.31-arm64.dmg
```
Setting Up LM Studio
Launch LM Studio and download the Llama 3.1 8B model from within the application.
Go to the Connect tab in LM Studio and configure the settings to obtain the LLM API Key.
Copy the API key and paste it into the chatbot's code (update the openai.api_key field).
Running the Project
Open your IDE or terminal and navigate to the project folder.
Ensure the JSON data file is located in the correct directory and update the path in the source file if necessary.
Execute the project code with the following command:
![image](https://github.com/user-attachments/assets/bd005955-a437-40bc-adb4-aa29ed7fa671)

Run the code
python Project.py file(Main File)
app.py and Mistral7B.py are secondary files which dont have extra features
![image](https://github.com/user-attachments/assets/424a7b52-8aaf-4ad7-ba73-e044951bd4e1)

Once the code runs, you will receive an HTTP server link (e.g., http://127.0.0.1:5000).
Open the link in your browser to access the InfoTech Assistant HTML webpage.
Establishing Connection
The chatbot interface will now display the InfoTech Assistant webpage where you can interact with the bot.
The backend establishes a connection between the chatbot, the LM Studio, and the JSON source file for query handling and response generation.
Key Features
Dynamic Scraping: Automates web scraping (using Selenium) to extract text and images from the InfoTechnology website.
Advanced NLP: Integrates SpaCy and Sentence Transformers for semantic understanding.
RAG Framework: Uses Retrieval-Augmented Generation for enhanced query processing.
Extensibility: Supports local storage and future cloud integration for scalability.
Termination
To stop the system:

Use the "bye" or "quit" command in the chatbot interface to terminate the session.
Manually stop the terminal running the backend server.

### Interactive HTML+JavaScript-based interface for the InfoTech Assistant chatbot. 
It allows users to interact with a backend chatbot API, fetch answers to their queries, and trigger data re-scraping.

#### Overview
The chatbot interface features:

A main heading linking to the InfoTechnology website.
A chatbot icon that toggles the chat window visibility.
A chat window for user interaction, where queries can be sent to the backend chatbot.
A button to trigger the data re-scraping process from the backend.
A responsive and modern design with a background image

#### Features
Dynamic Chat Window:

Users can toggle the chat window visibility by clicking the chatbot icon.
Allows sending queries to the backend via an input box and "send" button.

Backend Integration:

Sends user queries to the backend API ("http://127.0.0.1:5000/get_answer") for processing.
Displays the bot's response in the chat window.

Data Re-Scraping:

A dedicated button to trigger data scraping via the backend API (http://127.0.0.1:5000/scrape_data).
Displays success or error messages based on the scraping status (for the future reference)

Design:

Includes a background image (optional) and responsive styles for a professional look.
Chat messages are displayed with a simple scrollable layout.
Installation and Setup

#### Prerequisites
A working Python backend server (e.g., Flask server).
Properly configured backend APIs for /get_answer and /scrape_data.
A browser to render the HTML page.

Steps to Setup the Chatbot Interface

Download the Project Files:
Save the provided HTML code in a file named index.html.

Configure Backend APIs:
Ensure that your Flask backend is running and accessible at http://127.0.0.1:5000.

Launch the HTML Page:
Open the index.html file in any modern browser (e.g., Chrome, Edge).

Test the Chatbot:
Click the chatbot icon to toggle the chat window.
Enter a query and click the "send" button to get a response from the backend. As shown in the below Image
![image](https://github.com/user-attachments/assets/4741ab21-117c-4b41-a9c9-a383973b7039)

#### HTML Structure

Main Heading:

Displays the title "InfoTechnology" with a link to the InfoTechnology website

```
<div class="main-heading">
    <a href="https://infotechnology.fhwa.dot.gov/bridge/" target="_blank">InfoTechnology</a>
</div>
```

#### Chatbot Icon:
A clickable icon (🤖) toggles the visibility of the chat window
```
<div class="chatbot-icon" onclick="toggleChat()">🤖</div>
```
#### Chat Window:

Contains the chat messages, input box, and buttons for sending messages and triggering data scraping:
```
<div class="chat-window" id="chatWindow">
    <div class="chat-header">InfoTech Assistant</div>
    <div class="chat-body" id="chatBody"></div>
    <div class="chat-input">
        <input type="text" id="userInput" placeholder="Type your message here...">
        <button onclick="sendMessage()"> ➤ </button>
    </div>
    <button onclick="triggerRescraping()" style="margin: 10px;">Re-scrape Data</button>
</div>
```
#### CSS Styling

Body Styling:

Centers the content on the page and applies the background image:
```
body {
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
    background-image: url('<background-image-url>');
    background-size: cover;
    background-position: center;
}
```
#### Chat Window:
Styled as a floating box with rounded corners and flexible layout:

```
.chat-window {
    position: fixed;
    bottom: 80px;
    right: 30px;
    width: 400px;
    height: 600px;
    background-color: white;
    border: 2px solid #4682B4;
    border-radius: 10px;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
```

#### Send Message:

Sends the user input to the backend API and displays the response

```
async function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    const response = await fetch("http://127.0.0.1:5000/get_answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userInput })
    });
    const data = await response.json();
    document.getElementById("chatBody").appendChild(document.createTextNode(data.text));
}
```

### Main Code 
#### Importing lib

```
import json
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import openai
from flask import Flask, request, jsonify, render_template, session
import os
```

### Features

Dynamic Chatbot System:

Processes user queries and provides intelligent, context-aware responses.
Combines data retrieval from JSON with advanced LLM-based generative capabilities.

NLP Integration:

Uses SpaCy for keyword extraction and SentenceTransformer for semantic similarity.
Employs RAG techniques for combining retrieval and generation of responses.

Interactive Interface:

Flask-based backend with a front-end HTML page for user interactions.
Includes a feature to re-initiate data scraping.

Open LM Studio and download the Llama 3.1 8B model.
Copy the LLM API Key from LM Studio and update the openai.api_key in the code
```
openai.api_key = "lm-studio"

```
#### Configure the JSON Data

Place your pre-scraped JSON file (InfoBridge_scraped_data.json) in the appropriate folder.

Update the data_path in the code with the JSON file's path
```
chatbot = InteractiveRAGChatBot(data_path="C:/Users/ Filepath/filename.JSON")

```
### How to Run
Open a terminal in the project folder.
Run the Flask application: For this project its main filename is app.py
```
python .\app.py
```
Access the application by opening the URL in your browser:
```
http://127.0.0.1:5000
```

### Code Explanation
#### Backend Logic
##### Class: InteractiveRAGChatBot

Handles NLP, keyword extraction, data retrieval, and LLM-based response generation.

Key Methods:

handle_greetings: Processes common greetings (e.g., "hi", "hello", "bye").
```
def handle_greetings(self, user_input: str) -> str:
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
```
process_data: Loads the JSON data and organizes it into manageable chunks.
extract_keywords: Uses SpaCy to extract keywords from user input.
Using of the transformers:
```
class InteractiveRAGChatBot:
    def __init__(self, data_path: str):
        # Load JSON data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        # Initialize SpaCy model for keyword extraction and sentence segmentation
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize SentenceTransformer for encoding and set device to CPU
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True)
```

search_data: Finds relevant data chunks from the JSON file based on keyword matches.

#### Flask Routes:

/: Renders the front-end HTML page.
/get_answer: Handles user queries and generates responses.
/scrape_data: Initiates re-scraping of data (future implementation).

### How It Works
The user inputs a query via the HTML front-end.
The query is sent to the Flask backend (/get_answer) as a POST request.
```
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
        similarity_threshold = 0.5
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
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                messages=[{"role": "system", "content": "You are InfoTechnology Bridge Assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7, # for the probability distribution
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
```

### Response Generation:
Data Loading and Processing:

The JSON data is preloaded and processed into searchable chunks containing text and optional images.
Query Handling:

The chatbot checks for greetings and directly responds if a match is found.
If no greeting is detected, it extracts keywords using SpaCy and searches for matching content in the dataset.
Similarity Matching:

Calculates embeddings for the user query and dataset chunks using SentenceTransformer.
Compares embeddings using cosine similarity and selects the most relevant chunk.
Dynamic Source Links:

Generates a URL based on extracted keywords (e.g., bridge-maintenance for "Tell me about bridge maintenance").
Appends the generated URL to the chatbot's response.
LLM Integration:

Combines retrieved content with user input and sends the context to an LLM for additional processing or enhanced responses.
Handles errors gracefully and provides fallback messages if the LLM fails.
Session Management:

Maintains a history of the last 5 interactions to create contextual responses.
Error Handling
Data Not Found: Returns a default message if no relevant content is found in the dataset.
LLM Errors: Handles API connection issues or invalid responses with a fallback error message.
```
def generate_response(self, user_input: str) -> str:
    
        #Generate a response to user input, including content from JSON data and a source link.
        # Base URL for generating dynamic source links
        base_url = "https://infotechnology.fhwa.dot.gov/"
        default_link = f"{base_url}bridge/"  # Fallback source link
        # If the user input matches any greeting pattern, return a predefined response.
        # Handle greetings
        greeting_response = self.handle_greetings(user_input)
        if greeting_response:
            return greeting_response

        # Extract keywords and search JSON data for relevant content
        keywords = self.extract_keywords(user_input)
        matched_chunks = self.search_data(keywords)
        chunk_response = ""
        source_link = default_link  # Default source link

        # Define a similarity threshold and use JSON data if relevant
        similarity_threshold = 0.5 # Minimum similarity score to consider a match relevant
        if matched_chunks:
            # Encode the user input and matched chunks for similarity comparison
            query_embedding = self.embedding_model.encode([user_input], convert_to_numpy=True)
            chunk_texts = [chunk["text"] for chunk in matched_chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
            similarities = np.dot(chunk_embeddings, query_embedding.T).flatten() # Calculate similarity scores and find the best-matching chunk
            best_match_index = np.argmax(similarities)
            best_similarity_score = similarities[best_match_index]
            # If the best match is above the similarity threshold, prepare the chunk response
            if best_similarity_score >= similarity_threshold:
                best_chunk = matched_chunks[best_match_index]
                chunk_response = self.summarize_if_needed(best_chunk["text"], max_length=700)
                images = best_chunk.get("images", [])[:3]
                if images: # Append images if available in the chunk data
                    images_html = "<br>".join([f'<img src="{img}" alt="Related Image" width="300">' for img in images])
                    chunk_response += f"<br><br>{images_html}"

                # Generate source link dynamically based on keywords
                dynamic_path = "-".join(keywords).lower()
                source_link = f"{base_url}{dynamic_path}/"

        # Append source link to the chunk response
        if chunk_response:
            chunk_response += f"<br><br><strong>Source:</strong> <a href='{source_link}' target='_blank'>{source_link}</a>"

        # Generate a response from the LLM
        try: # Retrieve the conversation history from the session for context
            conversation_history = session.get('conversation_history', [])
            session_context = "\n".join(
                [f"User: {msg['content']}\nAssistant: {msg['content']}" for msg in conversation_history[-5:]]
            )# Combine session context and JSON context to form the prompt for the LLM
            prompt = f"{session_context}\n\nContext: {chunk_response}\nUser Question: {user_input}" if chunk_response else user_input
            # Call the LLM using the OpenAI-style API
            completion = openai.ChatCompletion.create(
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                messages=[{"role": "system", "content": "You are InfoTechnology Bridge Assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7, # Adjust the randomness of the model's response
            )
            llm_response = completion.choices[0].message['content'].strip()
        except Exception as e:
            print(f"Error connecting to LLM: {e}")
            llm_response = "I'm having trouble connecting to my language model. Please try again later."

        # Combine JSON chunk response and LLM response
        combined_response = (
            f"{chunk_response}<br><br><strong>LLM Response:</strong><br>{llm_response}" 
            if chunk_response else llm_response
        )

        # Update the session with the new conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": combined_response})
        session['conversation_history'] = conversation_history  # Store updated history in session
        session.modified = True  # Ensure Flask saves the session data
        #Return the final combined response
        return combined_response
```
#### Mistral 7B
For the Mistral 7B need to change the model name here as shown in the below code.
```
Generate a response from LLM using OpenAI-style API
        try:
            completion = openai.ChatCompletion.create(
                model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                messages=[{"role": "system", "content": "You are InfoTechnology Bridge Assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,
```

##### The chatbot:
Extracts keywords using SpaCy.
Searches the JSON file for relevant content.
Generates a response using both the JSON content and the LLM.
The response is returned to the front-end and displayed to the user.
Users can initiate data scraping via the /scrape_data route.

```
# Initialize the chatbot
chatbot = InteractiveRAGChatBot(data_path="C:/Users/samee/OneDrive/Desktop/InfoBridge_scraped_data/InfoBridge_scraped_data.json")

@app.route('/')#Flask app which enroute to the HTML file for input and output
def index():
    return render_template("index.html")

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.json.get("question", "")#to get the user input
    response = chatbot.generate_response(user_input)# post response to user
    return jsonify({"text": response})

@app.route('/scrape_data', methods=['POST'])
def scrape_data():
    return jsonify({"status": "Re-scraping initiated!"})#Rescraping initiation (Need best System Config that which can handle the process)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```


#### Front-End Interface
The chatbot includes a responsive front-end interface built with HTML, CSS, and JavaScript. Key features:

1.Chat Window: Allows users to type queries and view responses.
2.Re-scrape Button: Initiates data scraping for dynamic updates.

#### Key Features of the Chatbot Logic

##### Hybrid Approach:

Combines JSON-based retrieval with LLM-based generation to provide accurate and informative responses.

##### Session Management:

Maintains conversation history for contextual responses using Flask sessions.

##### Similarity Matching:

Uses SentenceTransformer to calculate semantic similarity between user queries and JSON chunks.

##### Flexible Scalability:

Can be extended to include cloud-based database storage and additional NLP pipelines.

#### Future Improvements
##### Enhanced Re-Scraping:

Automate the re-scraping process with dynamic scheduling and improved system configurations.
###### Cloud Integration:

Host the backend on cloud platforms for wider accessibility and scalability.
##### Mobile-Friendly Interface:

Make the HTML page responsive for mobile devices.
##### Error Handling:

Implement more robust error handling for API calls and missing data scenarios.


### Some Common Issues

LLM Connection Error:

Ensure LM Studio is running and the correct API key is configured.

Data Not Found:

Verify the JSON file path is correct and contains valid data.


### Response Evaluation
The provided code evaluates the similarity between the chatbot's actual responses and the expected responses for a set of test queries. It uses Sentence Transformers to compute semantic similarity scores and determines the correctness of the chatbot's responses based on a pre-defined similarity threshold.

#### Code Explanation
Dependencies:

json: To handle JSON-like data structures (test cases in this script).

numpy: Not directly used here but typically supports numerical operations in such contexts.

sentence_transformers: Provides pre-trained models for generating embeddings and utilities for calculating similarity scores.
Loading the SentenceTransformer Model:
```
embedding_model = SentenceTransformer("all-mpnet-base-v2")
```

Defining the Similarity Calculation Function:
```
def calculate_similarity(expected_response: str, actual_response: str) -> float:
    expected_embedding = embedding_model.encode(expected_response, convert_to_tensor=True)
    actual_embedding = embedding_model.encode(actual_response, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(expected_embedding, actual_embedding).item()
    return similarity_score
```
Converts expected_response and actual_response into embeddings using SentenceTransformer.encode().

Calculates the cosine similarity between the two embeddings using util.pytorch_cos_sim().

Returns a similarity score between 0.0 and 1.0, where 1.0 indicates identical semantics.

#### Test Cases:
```
test_cases = [
    {
        "query": "What are benefits of Hammer Sounding",
        "expected_response": """   """,
        "actual_response": " "
    },
]
```

The test cases are structured as dictionaries with the following keys:

query: The user input/query being tested.

expected_response: The ideal (expected) response.

actual_response: The chatbot's generated response.

#### Defining a Similarity Threshold:
```
similarity_threshold = 0.85
```
A threshold of 0.85 is used to determine whether the chatbot's response is "correct".
Higher thresholds indicate stricter requirements for correctness.

#### Evaluating Responses:
```
for case in test_cases:
    similarity_score = calculate_similarity(case["expected_response"], case["actual_response"])
    if similarity_score >= similarity_threshold:
        correct_count += 1
    print(f"Query: {case['query']}")
    print(f"Similarity Score: {similarity_score:.2f} - {'Correct' if similarity_score >= similarity_threshold else 'Incorrect'}\n")
```
For each test case, the function:
Computes the similarity score.
Compares the score with the threshold.
Updates the counter for correct responses (correct_count) if the score meets the threshold.
Outputs the query, similarity score, and whether the response is deemed "Correct" or "Incorrect".
### Calculating Accuracy:

```
accuracy = correct_count / len(test_cases) * 100
print(f"Overall Accuracy: {accuracy:.2f}%")
```
#### Set Up Test Cases:

Define your test cases as dictionaries in the test_cases list. Each test case should include:
query: The user input.
expected_response: The ideal response from the chatbot.
actual_response: The chatbot's actual output.

#### Adjust the Threshold:

Modify the similarity_threshold to tune the strictness of correctness (e.g., 0.85 for a high standard).
#### Run the Script:

Execute the script to evaluate the chatbot's performance. The output will display:
Query, similarity score, and whether it is correct or incorrect.
Overall accuracy of the chatbot for the given test cases.



#### Web Scraping for InfoBridge Content
This Python script uses Selenium to scrape data dynamically from the InfoTechnology InfoBridge website. The script retrieves content and images associated with various posts, identified by unique post_ids, and stores them in a structured JSON format.

Features
Dynamic Post ID Retrieval:

Automatically extracts all post_ids from the website in a single session.
Content Scraping:

For each post_id, retrieves the corresponding text content and image URLs by interacting with the website elements dynamically.
Image Handling:

Ensures that all image URLs from the expanded content are collected in sequential order, avoiding duplicates.
JSON Output:

Stores the scraped data in a JSON file with each post_id as a key.

Requirements
System Requirements
Browser: Google Chrome (ensure the correct version matches your chromedriver).
Operating System: Windows, macOS, or Linux.
Python Libraries
Selenium: For browser automation.
JSON: For storing data in a structured format.
time: For delays to ensure dynamic content is fully loaded.

How It Works
Retrieve All Post IDs:

Opens the InfoBridge website.
Identifies elements with the class get-single-page.
Extracts data-postid attributes from all such elements.
Scrape Content for Each Post:

For each post_id, reopens the website and identifies the corresponding element.
Simulates a click to expand the content using Selenium ActionChains.
Retrieves the text and image URLs from the expanded content.
Error Handling:

Logs errors during scraping (e.g., missing elements or timeout issues) to ensure the process continues for other post_ids.
Output:

```
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time

# Dictionary to store the scraped data
data = {}

# Step 1: Retrieve all `post_id`s dynamically in one session
driver = webdriver.Chrome()
driver.get('https://infotechnology.fhwa.dot.gov/bridge/') 

try:
    # Wait until all elements with `.get-single-page` are loaded
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "get-single-page")))  
    
    # Collect all `data-postid` attributes, ignoring elements without `data-postid`
    elements = driver.find_elements(By.CLASS_NAME, "get-single-page")
    post_ids = [element.get_attribute("data-postid") for element in elements if element.get_attribute("data-postid")]

finally:
    driver.quit()  # Close the initial browser after retrieving all post_ids

# Step 2: Loop through each `post_id` and reopen the browser for each to scrape content
for post_id in post_ids:
    driver = webdriver.Chrome()
    driver.get('https://infotechnology.fhwa.dot.gov/bridge/')  # Reload the website 

    try:
        # Wait until the element with the specific `data-postid` is present
        element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f".get-single-page[data-postid='{post_id}']"))
        )
        
        # Simulate click to expand the content
        ActionChains(driver).move_to_element(element).click(element).perform()

        # Allow time for AJAX to load expanded content
        time.sleep(3)

        # Locate the expanded content and retrieve text
        expanded_content = driver.find_element(By.ID, 'single-page-content') 
        text_content = expanded_content.text
        
        # Retrieve all image URLs within the expanded content in sequential order
        images = expanded_content.find_elements(By.TAG_NAME, 'img')
        image_urls = []
        for img in images:
            img_url = img.get_attribute('src')
            if img_url not in image_urls:  # Add only if it’s not already in the list
                image_urls.append(img_url)

        # Store the text and images in the dictionary with the post_id as the key
        data[post_id] = {
            "text": text_content,
            "images": image_urls  # Will be an empty list if no images are found
        }

    except Exception as e:#for error handling
        print(f"An error occurred for post_id {post_id}: {e}")
    finally:
        # Close the browser session after scraping each element
        driver.quit()

# Print and save data as JSON after all scraping is complete
print("Scraped Data:")
print(json.dumps(data, indent=4))  # for reference

# Save to a JSON file
with open("InfoBridge_scraped_data.json", "w") as file:
    json.dump(data, file, indent=4)

```

#### References
[1].Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., ... & Zhou, J.
(2023). How abilities in large language models are affected by supervised fine-tuning data composition. arXiv preprint arXiv:2310.05492.
[2].Perez, S. P., Zhang, Y., Briggs, J., Blake, C., Levy-Kramer, J., Balanca, P., ... & Fitzgibbon, A. W. (2023). Training and inference
of large language models using 8-bit floating point. arXiv preprint
arXiv:2309.17224.
[3]. Stack Overflow. "Using Selenium to click page and scrape info from routed page." Available: \url{https://stackoverflow.com/questions/71786531/using-selenium-to-click-page-and-scrape-info-from-routed-page}. Accessed: Nov. 02, 2024. 
