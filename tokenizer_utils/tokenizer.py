from typing import List
from vncorenlp import VnCoreNLP

class ViToeknizer(object):
    def __init__(self):
        self.annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')

    def tokenize(self, sentences: List[list]) -> List[list]:
        return self.annotator.tokenize(sentences)
