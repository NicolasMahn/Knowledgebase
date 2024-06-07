import gradio as gr
import yaml

import query_data


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config("config.yaml")
    data_topics = config['data_topics']
    default_topic = config['default_topic']

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--reset", action="store_true", help="Reset the database.")
    #parser.add_argument("--debug", action="store_true", help="Additional print statements")
    #parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    #args = parser.parse_args()

    #selected_topic = args.topic if args.topic else default_topic
    topic_config = data_topics[default_topic]
    topic_dir = topic_config['topic_dir']

    ChatClient(topic_dir).launch_chat_client()


class ChatClient:
    def __init__(self, topic_dir: str):
        self.topic_dir = topic_dir
        self.chroma_dir = f"{self.topic_dir}/chroma"
        # self.data_dir = f"{self.topic_dir}/documents"

    def chat_function(self, message: str, history: list):
        response, metadata_list = query_data.query_rag(message, self.chroma_dir, debug=True)

        answer = f"Rsponse:\n{response}\n\n"
        answer += f"Scources:\n"
        for i, metadata in enumerate(metadata_list):
            answer += f"   {i}. [{metadata['type']} | URL: {metadata['url']} ]\n"

        return answer

    def launch_chat_client(self):
        gr.ChatInterface(
            fn=self.chat_function,
            chatbot=gr.Chatbot(height=900, placeholder="Ask me any question about the Panda robot!"),
            title="Production Knowledgebase",
            description="Ask me any question about the Panda robot!",
            theme="soft",
            examples=["What is the Load Capacity of the Panda Robot?", "How much Force can the Panda Robot exert?",
                      "What can the Panda Robot be used for?"],
            cache_examples=True,
            retry_btn=None,
            undo_btn="Delete Previous",
            clear_btn="Clear"
        ).launch()


if __name__ == "__main__":
    main()
