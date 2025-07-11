from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

def generate_fake_news(prompt="Breaking News:"):
    output = generator(prompt, max_length=100, num_return_sequences=1)
    return output[0]['generated_text']
