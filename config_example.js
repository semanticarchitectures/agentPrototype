import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';

// Load environment variables
dotenv.config();

// Get API keys
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

// Validate required keys
if (!OPENAI_API_KEY) {
  throw new Error('OPENAI_API_KEY environment variable is required');
}

// Use in your code
const llm = new ChatOpenAI({
  openAIApiKey: OPENAI_API_KEY,
  modelName: 'gpt-4'
});
