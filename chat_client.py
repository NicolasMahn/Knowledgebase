import gradio as gr

def chat_function(message: str, history: list):
    # hier der aufruf zu Nicolas seinem Code

    return "Test response"

demo = gr.ChatInterface(
    fn=chat_function,
    chatbot=gr.Chatbot(height=400, placeholder="Ask me any question about the Panda robot!"),
    title="Production knowledgebase",
    description="Ask me any question about the Panda robot!",
    theme="soft",
    examples=["Test1", "Test2", "Test3"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear"
).launch()