import os
import unittest
from unittest.mock import patch

class TestOpenAIAPIKey(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_read_openai_api_key(self):
        # Import the module that reads the API key
        import openai
        
        # Read the API key from environment variables
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Assert that the API key was read correctly
        self.assertEqual(openai.api_key, "test_api_key")

if __name__ == "__main__":
    unittest.main()