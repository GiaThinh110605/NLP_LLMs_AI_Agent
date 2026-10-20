import ollama

img_path = "/Users/mac/Downloads/Learn_LLMs_AI_Agent/module_3/images.webp"

response = ollama.chat(
    model="qwen3.5:0.8b",
    messages=[
        {
            "role": "user",
            "content": "What animal in this image?",
            "images": [img_path]
        }
    ]
)

print(response.message.content)