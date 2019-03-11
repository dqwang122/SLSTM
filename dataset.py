#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from fastNLP.io.dataset_loader import DataSetLoader, convert_seq2seq_dataset


class MyConllLoader(DataSetLoader):
    """loader for conll format files"""

    def __init__(self):
        """
        :param str data_path: the path to the conll data set
        """
        super(MyConllLoader, self).__init__()

    def load(self, data_path):
        """
        :return: list lines: all lines in a conll file
        """
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        sentences = self.parse(lines)
        data = []
        for sent in sentences:
            word_seq = []
            tag_seq = []
            if(len(sent) == 1):
                continue
            for item in sent:
                word_seq.append(item[0])
                tag_seq.append(item[1])
            data.append([word_seq, tag_seq])
        return self.convert(data)

    @staticmethod
    def parse(lines):
        """
        :param list lines:a list containing all lines in a conll file.
        :return: a 3D list
        """
        sentences = list()
        tokens = list()
        for line in lines:
            if line[0] == "#":
                # skip the comments
                continue
            if line == "\n":
                sentences.append(tokens)
                tokens = []
                continue
            tokens.append(line.split())
        return sentences

    def convert(self, data):
        return convert_seq2seq_dataset(data)