from google import genai
from google.genai import types


class AI_Handler:
    def __init__(self, client, model, config):
        
        self.client = client
        self.model = model
        self.config = config
    def ask_llm(self, text):
        self.response = self.client.models.generate_content(
            # "gemini-2.5-flash-lite"
            # You are a helpful writing assistant. The user will give you a voice command. Respond with only the output text, no explanations.
            model= self.model,
            config = self.config,
            contents = text
        )
        return self.response.text