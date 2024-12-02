# Hugging Face LLM

from transformers import pipeline


class HuggingFaceLLM:
    def __init__(self):
        # Load facebook/bart-large-cnn summarization pipeline
        self.hugging_face_summarizer = pipeline("summarization", model= "facebook/bart-large-cnn")

    def summarize(self, input_text):
        hugging_face_summary = self.hugging_face_summarizer(input_text, max_length=54, min_length=32, do_sample=False)
        return f"Hugging Face Summary: {hugging_face_summary[0]['summary_text']}"
