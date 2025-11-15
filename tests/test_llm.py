import unittest
from unittest.mock import MagicMock, patch

from virtual_lab.agent import Agent
from virtual_lab.llm.groq import GroqClient
from virtual_lab.llm.anthropic import AnthropicClient
from virtual_lab.llm.gemini import GeminiClient
from virtual_lab.llm.ollama import OllamaClient
from virtual_lab.llm.openrouter import OpenRouterClient


class TestLLMClients(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(
            title="Test Agent",
            expertise="testing",
            goal="to test",
            role="tester",
            model="test-model",
        )

    @patch('groq.Groq')
    def test_groq_client_create_assistant(self, mock_groq):
        # Arrange
        client = GroqClient(api_key="test_key")

        # Act
        assistant = client.create_assistant(agent=self.agent, pubmed_search=True)

        # Assert
        self.assertEqual(assistant["name"], "Test Agent")
        self.assertEqual(assistant["model"], "test-model")
        self.assertEqual(len(assistant["tools"]), 1)
        self.assertEqual(assistant["tools"][0]['function']['name'], 'pubmed_search')

    @patch('anthropic.Anthropic')
    def test_anthropic_client_create_assistant(self, mock_anthropic):
        # Arrange
        client = AnthropicClient(api_key="test_key")

        # Act
        assistant = client.create_assistant(agent=self.agent, pubmed_search=True)

        # Assert
        self.assertEqual(assistant["name"], "Test Agent")
        self.assertEqual(assistant["model"], "test-model")
        self.assertEqual(len(assistant["tools"]), 1)
        self.assertEqual(assistant["tools"][0]['function']['name'], 'pubmed_search')

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_gemini_client_create_assistant(self, mock_configure, mock_gemini_model):
        # Arrange
        client = GeminiClient(api_key="test_key")

        # Act
        assistant = client.create_assistant(agent=self.agent, pubmed_search=True)

        # Assert
        self.assertEqual(assistant["name"], "Test Agent")
        self.assertEqual(assistant["model"], "test-model")
        self.assertEqual(len(assistant["tools"]), 1)
        self.assertEqual(assistant["tools"][0]['function']['name'], 'pubmed_search')

    @patch('ollama.Client')
    def test_ollama_client_create_assistant(self, mock_ollama):
        # Arrange
        client = OllamaClient()

        # Act
        assistant = client.create_assistant(agent=self.agent, pubmed_search=True)

        # Assert
        self.assertEqual(assistant["name"], "Test Agent")
        self.assertEqual(assistant["model"], "test-model")
        self.assertEqual(len(assistant["tools"]), 1)
        self.assertEqual(assistant["tools"][0]['function']['name'], 'pubmed_search')

    @patch('openai.OpenAI')
    def test_openrouter_client_create_assistant(self, mock_openai):
        # Arrange
        client = OpenRouterClient(api_key="test_key")

        # We need to mock the call to the superclass's create_assistant
        client.client = mock_openai.return_value

        # Act
        assistant = client.create_assistant(agent=self.agent, pubmed_search=True)

        # Assert
        client.client.beta.assistants.create.assert_called_once()
        call_args = client.client.beta.assistants.create.call_args
        self.assertEqual(call_args.kwargs["name"], "Test Agent")
        self.assertEqual(call_args.kwargs["model"], "test-model")
        self.assertEqual(len(call_args.kwargs["tools"]), 1)
        self.assertEqual(call_args.kwargs["tools"][0]['function']['name'], 'pubmed_search')


if __name__ == '__main__':
    unittest.main()
