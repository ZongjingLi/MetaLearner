import openai
import time
from openai.error import RateLimitError, Timeout, APIError, APIConnectionError, OpenAIError

#`pip install openai==0.28`
openai.api_key = "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"

def run_gpt(questions, prompts, temperature: float = 1.0, use_user_message: bool = False):
    query_str = '\n'.join([  # Format the questions into a query string
        '<text>{}</text>'.format(q) for q in questions
    ])

    response = None
    for i in range(10):  # Retry up to 10 times
        try:
            # Create a chat completion request
            response = openai.ChatCompletion.create(
                model='gpt-4o-mini',  # Specify the desired model (e.g., gpt-4o-mini or gpt-3.5-turbo)
                messages=[
                    {'role': 'system', 'content': prompts['system']},  # System message to set context
                    {'role': 'user', 'content': prompts['user'] + query_str}  # User message containing the prompt and questions
                ],
                max_tokens=1024,  # You can adjust the max tokens as needed
                temperature=temperature,  # Controls randomness
            )
            break  # If successful, exit the loop
        except RateLimitError:
            print('Rate limit exceeded, retrying...')
            time.sleep(30)  # Wait for 30 seconds before retrying
        except (APIError, APIConnectionError):
            print('API error, retrying...')
            time.sleep(30)  # Wait for 30 seconds before retrying

    # Ensure a response was received
    assert response is not None

    # Return the response in a structured format
    return {
        'questions': questions,
        'response': response['choices'][0]['message']['content'],  # Extract the content of the response
        'raw_response': response  # Optionally return the full response for debugging
    }


def llm_parse(questions):
    prompt = """core/prompt/metalearn_prompts.txt"""
    with open(prompt) as f:
        prompts_str = f.read()
        system_prmopt, user_prompt = prompts_str.split('----')
        prompts = {
            'system': system_prmopt.strip(),
            'user': user_prompt.strip()
        }

    
    res = run_gpt(questions, prompts)
    return res

if __name__  == "__main__":
    question = ["the red mob at right of you is minotaur", "the blue mob is fungus"]
    res = llm_parse(question)
    print(res["questions"])
    print(res["response"])