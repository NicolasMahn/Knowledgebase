import gradio as gr

import query_data

def chat_function(message: str, history: list):
    # hier der aufruf zu Nicolas seinem Code
    topic_dir = "data/franka_emika_panda"
    chroma_dir = f"{topic_dir}/chroma"
    data_dir = f"{topic_dir}/documents"
    response = query_data.query_rag(message, data_dir, chroma_dir)

    return response

demo = gr.ChatInterface(
    fn=chat_function,
    chatbot=gr.Chatbot(height=400, placeholder="Ask me any question about the Panda robot!"),
    title="Production knowledgebase",
    description="Ask me any question about the Panda robot!",
    theme="soft",
    examples=["Test1"],
    cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear"
).launch()