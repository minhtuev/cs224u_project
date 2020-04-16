import os
import xml.etree.ElementTree as ET


dataset = ['DrugBank', 'MedLine']


def read_xml_file(file_path):
    print(file_path)
    return ET.parse(file_path).getroot()


class Entity:
    def __init__(self, entity_id, char_offset=None, type=None, text=None):
        self._id = entity_id
        self.char_offset = char_offset
        self.type = type
        self.text = text

    @staticmethod
    def from_xml(xml_object):
        entity = Entity(
            xml_object.attrib['id'],
            char_offset=xml_object.attrib['charOffset'],
            type=xml_object.attrib['type'],
            text=xml_object.attrib['text']
        )
        return entity


class Sentence:
    def __init__(self, sent_id, text=None, entities=None):
        self._id = sent_id
        self.text = text or ''
        self.entities = entities or []
        self.map = {}

    @staticmethod
    def from_xml(xml_object):
        sent = Sentence(xml_object.attrib['id'], text=xml_object.attrib['text'])
        entity_count = 0
        for entity in xml_object.iter('entity'):
            sent.entities.append(Entity.from_xml(entity))
            entity_count += 1

        pair_count = 0
        for pair in xml_object.iter('pair'):
            if pair.attrib['ddi'] == 'true':
                e1 = pair.attrib['e1']
                e2 = pair.attrib['e2']
                mechanism = pair.attrib['type']
                sent.map[e1] = (e2, mechanism)
            pair_count += 1

        if pair_count != entity_count*(entity_count-1)/2:
            print('Potential data issue for sentence ', sent._id)

        return sent


class Document:
    def __init__(self, doc_id, sentences=None):
        self._id = doc_id
        self.sentences = sentences or []

    @staticmethod
    def read_from_xml(filepath):
        return Document.from_xml(read_xml_file(filepath))

    @staticmethod
    def from_xml(xml_object):
        doc = Document(xml_object.attrib['id'])
        for sentence in xml_object.iter('sentence'):
            doc.sentences.append(Sentence.from_xml(sentence))
        return doc


class Dataset:
    def __init__(self, name, documents=None):
        self.name = name
        self.documents = documents or []

    @staticmethod
    def from_training_data(name):
        ds = Dataset(name)
        path = "./Train/" + name
        for _, _, files in os.walk(path, topdown=False):
            print(files)
            for filename in files:
                filepath = path + '/' + filename
                if '.xml' in filename:
                    ds.documents.append(Document.read_from_xml(filepath))
        return ds
