from transformers import pipeline

from elg import FlaskService
from elg.model import ClassificationResponse #AnnotationsResponse
# requires python modules: docker flask requests_toolbelt

class SlobertaTweetsentiment(FlaskService):

    nlp = pipeline("sentiment-analysis", "sloberta-tweetsentiment", use_fast=False)

    def convert_outputs(self, outputs, content):
        classes = []
        for i in range(len(outputs)):
            sentence = content[i]
            label = outputs[i]["label"]
            score = outputs[i]["score"]
            classes.append({"class": label, "score": score})
        return ClassificationResponse(classes=classes)

    def process_text(self, content):
        outputs = self.nlp(content.content)
        return self.convert_outputs(outputs, content.content)

flask_service = SlobertaTweetsentiment("sloberta-sentiment")
app = flask_service.app
