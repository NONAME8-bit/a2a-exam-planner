import gradio as gr
import requests

API_URL = "http://localhost:8000/chat"  # FastAPI backend

def chat_with_host_agent(message, history):
    try:
        response = requests.post(
            API_URL,
            json={"query": message, "session_id": "gradio-session"},
            timeout=30,
        )
        data = response.json()
        if "content" in data and data["content"]:
            return history + [[message, data["content"]]]
        elif "updates" in data:
            return history + [[message, data["updates"]]]
        else:
            return history + [[message, "(no response)"]]
    except Exception as e:
        return history + [[message, f"Error: {e}"]]

with gr.Blocks() as demo:
    gr.Markdown("#  Host Agent Chat")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask Host Agent")
    clear = gr.Button("Clear Chat")

    def user_input(message, history):
        return "", chat_with_host_agent(message, history)

    msg.submit(user_input, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)

