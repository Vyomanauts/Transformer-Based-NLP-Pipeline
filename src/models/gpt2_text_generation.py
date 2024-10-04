from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Ensure the tokenizer's padding token ID is set (to avoid warnings)
tokenizer.pad_token = tokenizer.eos_token

# Function to generate text based on a prompt
def generate_text(prompt, max_length=100):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7,
        do_sample=True,  # Enable sampling-based generation
        pad_token_id=tokenizer.eos_token_id  # Use eos_token_id as pad_token_id
    )
    
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example: Generate text based on a prompt
prompt = "The movie was fantastic because"
generated_text = generate_text(prompt, max_length=50)
print(generated_text)
