import os
import json
import re
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

class CustomTransformation:
    def __call__(self, documents):
        transformed_documents = []
        for doc in documents:
            transformed_content = doc.get_content().lower()
            transformed_content = re.sub(r'\s+', ' ', transformed_content)
            transformed_content = re.sub(r'[^\w\s]', '', transformed_content)
            transformed_documents.append(Document(text=transformed_content, metadata=doc.metadata))
        return transformed_documents


def Sentence_Splitter_docs_into_nodes(all_documents):
    try:
        splitter = SentenceSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        nodes = splitter.get_nodes_from_documents(all_documents)

        return nodes

    except Exception as e:
        print(f"Error splitting documents into nodes: {e}")
        return []

def save_nodes(nodes, output_file):
    try:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Convert the TextNode objects to dictionaries
        nodes_dict = [node.dict() for node in nodes]

        with open(output_file, 'w') as file:
            json.dump(nodes_dict, file, indent=4)
        print(f"Saved nodes to {output_file}")
    except Exception as e:
        print(f"Error saving nodes to file: {e}")


if __name__ == '__main__':
    try:
        # Load data from directory
        documents = SimpleDirectoryReader(input_dir=r"\data").load_data()
        print(f"Loaded {len(documents)} documents")

        if documents:
            # Apply custom transformation
            custom_transform = CustomTransformation()
            documents = custom_transform(documents)

            # Split documents into nodes
            nodes = Sentence_Splitter_docs_into_nodes(documents)

            print(f"Created {len(nodes)} nodes")

            # Save nodes to a single JSON file
            output_file = r"\data\nodes.json"
            save_nodes(nodes, output_file)

        else:
            print("No documents to process.")

    except Exception as e:
        print(f"Error processing documents: {e}")
