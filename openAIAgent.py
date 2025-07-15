from dotenv import load_dotenv
import os
import openai

class OpenAIAgent:
    def __init__(self, api_key = None):
        self.api_key = api_key
        if load_dotenv(): # load .env file to get API KEY
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            api_key = input("Please enter your API key: ")
            openai.api_key = api_key
        
        # Check if API key is loaded
        if not openai.api_key:
            raise ValueError("API Key not found.")
        
        self.client = openai.OpenAI(api_key=openai.api_key)
        
    def test_api(self):
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": "Hello from OpenAI API!"}]
        )
        return response.choices[0].message.content
    