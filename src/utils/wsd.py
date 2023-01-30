from pathlib import Path

from conllu import parse_incr
import tempfile
import xml.etree.cElementTree as ET
from typing import NamedTuple, Optional, List, Callable, Tuple, Iterable

from src.utils.commons import execute_bash_command

pos_map = {
    # U-POS
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "PROPN": "n",
    # PEN
    "AFX": "a",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "MD": "v",
    "NN": "n",
    "NNP": "n",
    "NNPS": "n",
    "NNS": "n",
    "RB": "r",
    "RP": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


class AnnotatedToken(NamedTuple):
    idx: int
    text: str
    pos: Optional[str] = None
    lemma: Optional[str] = None


class WSDInstance(NamedTuple):
    annotated_token: AnnotatedToken
    labels: Optional[List[str]]
    instance_id: Optional[str]


def read_from_raganato(
    xml_path: str,
    key_path: Optional[str] = None,
    instance_transform: Optional[Callable[[WSDInstance], WSDInstance]] = None,
) -> Iterable[Tuple[str, str, List[WSDInstance]]]:
    def read_by_text_iter(xml_path: str):

        it = ET.iterparse(xml_path, events=("start", "end"))
        _, root = next(it)

        for event, elem in it:
            if event == "end" and elem.tag == "text":
                document_id = elem.attrib["id"]
                for sentence in elem:
                    sentence_id = sentence.attrib["id"]
                    for word in sentence:
                        yield document_id, sentence_id, word

            root.clear()

    mapping = {}

    if key_path is not None:
        with open(key_path) as f:
            for line in f:
                line = line.strip()
                wsd_instance, *labels = line.split(" ")
                mapping[wsd_instance] = labels

    last_seen_document_id = None
    last_seen_sentence_id = None

    for document_id, sentence_id, element in read_by_text_iter(xml_path):

        if last_seen_sentence_id != sentence_id:

            if last_seen_sentence_id is not None:
                yield last_seen_document_id, last_seen_sentence_id, sentence

            sentence = []
            last_seen_document_id = document_id
            last_seen_sentence_id = sentence_id

        annotated_token = AnnotatedToken(
            idx=len(sentence),
            text=element.text,
            pos=element.attrib.get("pos", None),
            lemma=element.attrib.get("lemma", None),
        )

        wsd_instance = WSDInstance(
            annotated_token=annotated_token,
            labels=None
            if element.tag == "wf" or element.attrib["id"] not in mapping
            else mapping[element.attrib["id"]],
            instance_id=None if element.tag == "wf" else element.attrib["id"],
        )

        if instance_transform is not None:
            wsd_instance = instance_transform(wsd_instance)

        sentence.append(wsd_instance)

    yield last_seen_document_id, last_seen_sentence_id, sentence


def expand_raganato_path(path: str) -> Tuple[str, str]:
    return f"{path}.data.xml", f"{path}.gold.key.txt"


class RaganatoBuilder:
    def __init__(self, lang: Optional[str] = None, source: Optional[str] = None):
        self.corpus = ET.Element("corpus")
        self.current_text_section = None
        self.current_sentence_section = None
        self.gold_senses = []

        if lang is not None:
            self.corpus.set("lang", lang)

        if source is not None:
            self.corpus.set("source", source)

    def open_text_section(self, text_id: str, text_source: str = None):
        text_section = ET.SubElement(self.corpus, "text")
        text_section.set("id", text_id)
        if text_source is not None:
            text_section.set("source", text_source)
        self.current_text_section = text_section

    def open_sentence_section(self, sentence_id: str, update_id: bool = True):
        sentence_section = ET.SubElement(self.current_text_section, "sentence")
        if update_id:
            sentence_id = self.compute_id([self.current_text_section.attrib["id"], sentence_id])
        sentence_section.set("id", sentence_id)
        self.current_sentence_section = sentence_section

    def add_annotated_token(
        self,
        token: str,
        lemma: str,
        pos: str,
        instance_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        update_id: bool = False,
    ):
        if instance_id is not None:
            token_element = ET.SubElement(self.current_sentence_section, "instance")
            if update_id:
                instance_id = self.compute_id([self.current_sentence_section.attrib["id"], instance_id])
            token_element.set("id", instance_id)
            if labels is not None:
                self.gold_senses.append((instance_id, " ".join(labels)))
        else:
            token_element = ET.SubElement(self.current_sentence_section, "wf")
        token_element.set("lemma", lemma)
        token_element.set("pos", pos)
        token_element.text = token

    @staticmethod
    def compute_id(chain_ids: List[str]) -> str:
        return ".".join(chain_ids)

    def store(self, data_output_path: str, labels_output_path: str):
        self.__store_xml(data_output_path)
        self.__store_labels(labels_output_path)

    def __store_xml(self, output_path: str):
        corpus_writer = ET.ElementTree(self.corpus)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/tmp.xml", "wb") as f_xml:
                corpus_writer.write(f_xml, encoding="UTF-8", xml_declaration=True)
            execute_bash_command(f" xmllint --format {tmp_dir}/tmp.xml > {output_path}")

    def __store_labels(self, output_path: str):
        with open(output_path, "w") as f_labels:
            for gold_sense in self.gold_senses:
                f_labels.write(" ".join(gold_sense))
                f_labels.write("\n")


def get_amalgum_lemma(conllu_token: dict, verb2compound_particle: dict, sense_intentory: "SenseInventory", sentence) -> Optional[str]:
    # particles in phrasal verbs like shut down should be omitted
    # if the words are next to each other
    if (
            conllu_token["deprel"] == "compound:prt" and
            conllu_token["id"] == conllu_token["head"] + 1
    ):
        phrasal_verb_lemma = f"{sentence[conllu_token['head']-1]['lemma']}_{conllu_token['lemma']}"
        if sense_intentory.get_possible_senses(phrasal_verb_lemma,  "v"):
            return None
        return conllu_token["lemma"]

    conllu_token_id = conllu_token["id"]
    compound_particle_token_id = verb2compound_particle.get(conllu_token_id)
    pos = conllu_token["upos"]
    if compound_particle_token_id is not None:
        compound_particle_lemma = sentence[compound_particle_token_id-1]["lemma"]
        lemma = f"{conllu_token['lemma']}_{compound_particle_lemma}"
        return lemma

    lemma = conllu_token["lemma"]
    postprocessed_lemma = conllu_token["lemma"].replace("-", "_")
    if sense_intentory.get_possible_senses(lemma, pos_map.get(pos, pos)):
        return lemma
    if sense_intentory.get_possible_senses(postprocessed_lemma, pos_map.get(pos, pos)):
        return postprocessed_lemma
    return lemma


def get_verb2compound_particle(sentence) -> dict:
    verb2compound_particle = {}
    ignored_heads = []
    for token in reversed(sentence):
        if token["deprel"] == "compound:prt":
            head_verb_token_id = token["head"]
            if head_verb_token_id in ignored_heads:
                assert head_verb_token_id not in verb2compound_particle
                continue
            if head_verb_token_id in verb2compound_particle:
                # ignore verbs with two or more particles
                del verb2compound_particle[head_verb_token_id]
                ignored_heads.append(head_verb_token_id)
                continue
            verb2compound_particle[head_verb_token_id] = token["id"]
    return verb2compound_particle

def _read_from_amalgum_document(
    document_path: Path, sense_intentory: "SenseInventory"
) -> Iterable[Tuple[str, str, List[WSDInstance]]]:
    document_id = document_path.stem
    with document_path.open("r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            verb2compound_particle = get_verb2compound_particle(sentence)
            wsd_instances = []
            sentence_id = sentence.metadata["sent_id"]
            for token in sentence:
                token_id = token["id"]
                lemma = get_amalgum_lemma(token, verb2compound_particle, sense_intentory, sentence)
                if lemma is None:
                    continue
                pos = token["upos"]
                annotated_token = AnnotatedToken(
                    idx=token_id,
                    text=token["form"],
                    pos=pos,
                    lemma=lemma,
                )
                instance_id = f"{document_id}__{sentence_id}__{token_id}"
                wsd_instance = WSDInstance(
                    annotated_token=annotated_token,
                    labels=None,
                    instance_id=instance_id if sense_intentory.get_possible_senses(lemma,
                                                                                   pos_map.get(pos, pos)) else None,
                )
                wsd_instances.append(wsd_instance)
            yield document_id, sentence_id, wsd_instances


def read_from_amalgum(
    amalgum_path: str,
    sense_intentory: "SenseInventory"
) -> Iterable[Tuple[str, str, List[WSDInstance]]]:
    if Path(amalgum_path).suffix == ".conllu":
        document_path = Path(amalgum_path)
        yield from _read_from_amalgum_document(document_path, sense_intentory)
    else:
        for genre_path in Path(amalgum_path).iterdir():
            for document_path in (genre_path / "dep").iterdir():
                yield from _read_from_amalgum_document(document_path, sense_intentory)
