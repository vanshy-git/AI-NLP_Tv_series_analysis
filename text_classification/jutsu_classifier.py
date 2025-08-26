import pandas as pd
import torch
import huggingface_hub
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          pipeline
            )
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
import gc
import sys, os

from text_classification.cleaner import Cleaner
import gradio as gr
from theme_classifier.theme_classifier import ThemeClassifier

from character_network import NamedEntityRecognizer, CharacterNetworkGenerator

from character_chatbot import CharacterChatBot
import os
from dotenv import load_dotenv
load_dotenv()

def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path,save_path)

    # Remove dialogue from the theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']

    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )

    return output_chart

def get_character_network(subtitles_path,ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path,ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

def classify_text(text_classifcation_model,text_classifcation_data_path,text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path = text_classifcation_model,
                                       data_path = text_classifcation_data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    output = output[0]
    
    return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatBot("AbdullahTarek/Naruto_Llama-3-8B",
                                         huggingface_token = os.getenv('huggingface_token')
                                         )

    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output


def main():
    with gr.Blocks() as iface:
        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Claasifiers)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Subtitles or script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button =gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list,subtitles_path,save_path], outputs=[plot])

        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtutles or Script Path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])

        # Text Classification with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output")
                    with gr.Column():
                        text_classifcation_model = gr.Textbox(label='Model Path')
                        text_classifcation_data_path = gr.Textbox(label='Data Path')
                        text_to_classify = gr.Textbox(label='Text input')
                        classify_text_button = gr.Button("Clasify Text (Jutsu)")
                        classify_text_button.click(classify_text, inputs=[text_classifcation_model,text_classifcation_data_path,text_to_classify], outputs=[text_classification_output])

        # Character Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(chat_with_character_chatbot)

    iface.launch(share=True)
            

if __name__ == '__main__':
    main()
from text_classification.training_utils import get_class_weights, compute_metrics

from text_classification.custom_trainer import CustomTrainer

class JutsuClassifier():
    def __init__(self,
                 model_path,
                 data_path=None,
                 text_column_name='text',
                 label_column_name='jutsu',
                 model_name = "distilbert/distilbert-base-uncased",
                 test_size=0.2,
                 num_labels=3,
                 huggingface_token = None
                 ):
        
        self.model_path = model_path
        self.data_path = data_path
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.model_name = model_name
        self.test_size = test_size
        self.num_labels = num_labels
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.huggingface_token = huggingface_token
        if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)
        
        self.tokenizer = self.load_tokenizer()

        if not huggingface_hub.repo_exists(self.model_path):

            # check if the data path is provided
            if data_path is None:
                raise ValueError("Data path is required to train the model,since the model path does not exist in huggingface hub")

            train_data, test_data = self.load_data(self.data_path)
            train_data_df = train_data.to_pandas()
            test_data_df = test_data.to_pandas()

            all_data = pd.concat([train_data_df, test_data_df]).reset_index(drop=True)
            class_weights = get_class_weights(all_data)

            self.train_model(train_data, test_data, class_weights)

        self.model = self.load_model(self.model_path)

    def load_model(self,model_path):
        model = pipeline('text-classification', model=model_path, return_all_scores=True)
        return model

    def train_model(self, train_data,test_data,class_weights):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict,
                                                                   )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir = self.model_path,
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            push_to_hub=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset = train_data,
            eval_dataset = test_data,
            tokenizer = self.tokenizer,
            data_collator=data_collator,
            compute_metrics= compute_metrics
        )

        trainer.set_device(self.device)
        trainer.set_class_weights(class_weights)

        trainer.train()

        # Flush Memory
        del trainer,model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def simplify_jutsu(self, jutsu):
        if "Genjutsu" in jutsu:
            return "Genjutsu"
        if "Ninjutsu" in jutsu:
            return "Ninjutsu"
        if "Taijutsu" in jutsu:
            return "Taijutsu"
    
    def preprocess_function(self,tokenizer,examples):
        return tokenizer(examples['text_cleaned'],truncation=True)

    def load_data(self,data_path):
        df = pd.read_json(data_path,lines=True)
        df['jutsu_type_simplified'] = df['jutsu_type'].apply(self.simplify_jutsu)
        df['text'] = df['jutsu_name'] + ". " + df['jutsu_description']
        df[self.label_column_name] = df['jutsu_type_simplified']
        df = df[['text', self.label_column_name]]
        df = df.dropna()

        # Clean Text
        cleaner = Cleaner()
        df['text_cleaned'] = df[self.text_column_name].apply(cleaner.clean)

        # Encode Labels 
        le = preprocessing.LabelEncoder()
        le.fit(df[self.label_column_name].tolist())

        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df['label'] = le.transform(df[self.label_column_name].tolist())

        # Train / Test Split
        test_size = 0.2
        df_train, df_test = train_test_split(df, 
                                            test_size=test_size, 
                                            stratify=df['label'],)
        
        # Conver Pandas to a hugging face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        # tokenize the dataset
        tokenized_train = train_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        tokenized_test = test_dataset.map(lambda examples: self.preprocess_function(self.tokenizer, examples),
                                            batched=True)
        
        return tokenized_train, tokenized_test

    def load_tokenizer(self):
        if huggingface_hub.repo_exists(self.model_path):
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def postprocess(self,model_output):
        output=[]
        for pred in model_output:
            label = max(pred, key=lambda x: x['score'])['label']
            output.append(label)
        return output

    def classify_jutsu(self,text):
        model_output = self.model(text)
        predictions =self.postprocess(model_output)
        return predictions