
from utils.trec_car_tools import iter_pages, Para, ParaText, ParaLink, Section, Image, List
from protocol_buffers.document_pb2 import Document, DocumentContent, EntityLink,  EntityLinkTotal

from nltk.stem import PorterStemmer
from string import punctuation

import stream
import copy
import time


class TrecCarParser:
    """ Class to parse TREC CAR cbor files to protocol buffer Document message (protocol_buffers/document_pb2/Document)
    """

    def __init__(self):

        # Initialise empty instance attribute for document.
        self.document = None
        self.nlp = None
        self.linking_metadata = None
        self.stemmer = None

    def get_manual_entity_links_from_para_bodies(self, bodies, text):
        """ Extract list manual entity_links (protocol_buffers/document.proto:EntityLink) from trec-car-tools
        Para.paragraph.bodies. """
        # Initialise counter to keep track of beginning of character of text span.
        text_start_i = 0
        manual_entity_links = []
        for b in bodies:

            if isinstance(b, ParaText):
                # Add the number of characters within body object to 'text_start_i'.
                text_start_i += len(b.get_text())

            elif isinstance(b, ParaLink):
                # 'text_end_i' calculated by adding the number of characters within anchor text to 'text_start_i'.
                text_end_i = text_start_i + len(b.get_text())

                # Create new EntityLink.AnchorTextLocation message for EntityLink.anchor_text_location.
                anchor_text_location = EntityLink.AnchorTextLocation()
                anchor_text_location.start = text_start_i
                anchor_text_location.end = text_end_i

                # Create new EntityLink message.
                manual_entity_link = EntityLink()
                manual_entity_link.anchor_text = b.anchor_text
                manual_entity_link.entity_id = b.pageid
                manual_entity_link.entity_name = b.page
                manual_entity_link.anchor_text_location.MergeFrom(anchor_text_location)

                # Append entity_link instance to entity_links.
                manual_entity_links.append(manual_entity_link)

                # Assertion ensures offsets are correct.
                assert text[text_start_i:text_end_i] == b.anchor_text
                # 'text_start_i' becomes end of anchor text span.
                text_start_i += len(b.get_text())

            else:
                print("Unexpected type: {}".format(type(b)))
                raise

        return manual_entity_links


    def __parse_skeleton_subclass_document_contents(self, skeleton_subclass, section_heading_ids, section_heading_names):
        """ Extract document_contents features (protocol_buffers/document.proto:DocumentContent) from trec-car-tools
        PageSkeleton subclass objects (List, Para, Section, or Image). """
        if isinstance(skeleton_subclass, Para):
            # Para features.
            content_id = skeleton_subclass.paragraph.para_id
            content_type = DocumentContent.ContentType.PARAGRAPH
            text = skeleton_subclass.paragraph.get_text()
            manual_entity_links = self.get_manual_entity_links_from_para_bodies(bodies=skeleton_subclass.paragraph.bodies, text=text)
            content_urls = []
            list_level = 0

        elif isinstance(skeleton_subclass, Image):
            # Image features.
            content_id = ''
            content_type = DocumentContent.ContentType.IMAGE
            text = ''
            manual_entity_links = []
            content_urls = [skeleton_subclass.imageurl]
            list_level = 0

            for caption in skeleton_subclass.caption:
                self.__parse_skeleton_subclass_document_contents(skeleton_subclass=caption,
                                                                 section_heading_ids=section_heading_ids,
                                                                 section_heading_names=section_heading_names)

        elif isinstance(skeleton_subclass, List):
            # List features.
            content_id = skeleton_subclass.body.para_id
            content_type = DocumentContent.ContentType.LIST
            text = skeleton_subclass.get_text()
            manual_entity_links = self.get_manual_entity_links_from_para_bodies(bodies=skeleton_subclass.body.bodies,
                                                                                text=text)
            content_urls = []
            list_level = skeleton_subclass.level


        elif isinstance(skeleton_subclass, Section):
            # Section features.
            section_heading_ids.append(skeleton_subclass.headingId)
            section_heading_names.append(skeleton_subclass.heading)
            children = skeleton_subclass.children
            # Loop over Section.children to parse nested PageSkeleton objects.
            for child in children:
                # There could be multiple nestings that will be handled.
                # 'section_heading_ids' and 'section_heading_names' are updated with nested levels.
                self.__parse_skeleton_subclass_document_contents(skeleton_subclass=child,
                                                                 section_heading_ids=copy.deepcopy(section_heading_ids),
                                                                 section_heading_names=copy.deepcopy(section_heading_names))
            # Return when completed parsing children.
            return

        else:
            print("NOT VALID skeleton_subclass type: {}".format(type(skeleton_subclass)))
            return

        # Build DocumentContent message.
        document_contents = DocumentContent()
        document_contents.section_heading_ids.extend(copy.deepcopy(section_heading_ids))
        document_contents.section_heading_names.extend(copy.deepcopy(section_heading_names))
        document_contents.content_type = content_type
        document_contents.content_id = content_id
        document_contents.text = text
        document_contents.manual_entity_links.extend(manual_entity_links)
        document_contents.content_urls.extend(content_urls)
        document_contents.list_level = list_level

        # Update Document message.
        self.document.document_contents.append(document_contents)


    def __add_document_contents_from_page_skeleton(self, skeleton):
        """ Loop over Page.skeleton appending contents to Document.document_contents. """
        for skeleton_subclass in skeleton:
            # Parse each PageSkeleton object in skeleton list.
            self.__parse_skeleton_subclass_document_contents(skeleton_subclass=skeleton_subclass,
                                                             section_heading_ids=[],
                                                             section_heading_names=[])

    def format_string(self, string):
        """ Format string - remove beginning/ending punctuation and stem. """
        return self.stemmer.stem(string.strip(punctuation))

    def __process_match_words(self, match_string):
        """ Process string of anchor text / entity name to list of word stems i.e. "Iain loves IR" -> ["iain", "lov",
        "ir"]. """
        process_match_words = []
        match_words = match_string.lower().split(" ")
        for w in match_words:
            w_stem = self.format_string(w)
            if len(w_stem) > 0:
                process_match_words.append(w_stem)
        return process_match_words


    def __get_entity_id_linking_metadata(self):
        """ Build require entity metadata for entity linking i.e. entity name mapping & stem tokens of anchor_text/
        entity_name for string matching process. """
        # Dict mapping entity_ids -> entity_names.
        entity_ids_to_entity_names = {}
        # Dict mapping entity_ids -> list of lists that contain valid word matching chunks produced from anchor text
        # and entity names.
        entity_ids_to_match_words_lists = {}
        for document_content in self.document.document_contents:
            for manual_entity_link in document_content.manual_entity_links:
                # If new entity_id:
                #    (1) Add entity_name -> entity_name in entity_ids_to_entity_names entity name[entity_name].
                #    (2) Add anchor_text and entity_id to entity_ids_to_match_words_lists[entity_name].
                entity_id = manual_entity_link.entity_id
                if entity_id not in entity_ids_to_entity_names:
                    # Add entity_name to entity_ids_to_entity_names map.
                    entity_ids_to_entity_names[entity_id] = manual_entity_link.entity_name
                    # Add anchor_text to entity_ids_to_match_words_lists[entity_name].
                    match_words_anchor_text = self.__process_match_words(match_string=manual_entity_link.anchor_text)
                    entity_ids_to_match_words_lists[entity_id] = []
                    if len(match_words_anchor_text) > 0:
                        entity_ids_to_match_words_lists[entity_id].append(match_words_anchor_text)
                    # Add entity_name to entity_ids_to_match_words_lists[entity_name].
                    if manual_entity_link.entity_name != manual_entity_link.anchor_text:
                        match_words_entity_name = self.__process_match_words(match_string=manual_entity_link.entity_name)
                        if (match_words_entity_name != match_words_anchor_text) and (len(match_words_entity_name) > 0):
                            entity_ids_to_match_words_lists[entity_id].append(match_words_entity_name)
                else:
                    # Add unique anchor_text to entity_ids_to_match_words_lists[entity_name].
                    match_words_anchor_text = self.__process_match_words(match_string=manual_entity_link.anchor_text)
                    if (match_words_anchor_text not in entity_ids_to_match_words_lists[entity_id]) \
                            and (len(match_words_anchor_text) > 0):
                        entity_ids_to_match_words_lists[entity_id].append(match_words_anchor_text)

        return entity_ids_to_entity_names, entity_ids_to_match_words_lists


    def __add_synthetic_entity_links(self):
        """ Add synthetic links to all document contents. """
        # If stemmer not initialised as field.
        if self.stemmer == None:
            self.stemmer = PorterStemmer()

        # Build entity metadata for entity linking.
        entity_ids_to_entity_names, entity_ids_to_match_words_lists = self.__get_entity_id_linking_metadata()

        for document_content in self.document.document_contents:
            text = document_content.text
            text_lower = document_content.text.lower()
            # Search for links for each entity id and words stems we are matching
            for entity_id, match_words_lists in entity_ids_to_match_words_lists.items():
                for match_words in match_words_lists:
                    # assert all word stems are in text_lower to avoid unnecessary searching.
                    if all(word in text_lower for word in match_words):
                        # Store index of text and text_lower for anchor_text locations.  Lower is required for edge
                        # cases where "AbCs".lower() != "AbCs".
                        start_i = 0
                        start_i_lower = 0
                        end_i = 0
                        end_i_lower = 0
                        # Indicates number of words matched (also acts as index of match_words).
                        matches = 0
                        # Required words matches to create entity link.
                        required_matches = len(match_words)
                        # For each word in text
                        for word_text in text.split(" "):
                            # Update character indexing.
                            end_i += len(word_text)
                            # Find lower of word for lower indexing.
                            word_text_lower = word_text.lower()
                            end_i_lower += len(word_text_lower)

                            # Access 'word_match' i.e. word stem we are trying to match.
                            word_match = match_words[matches]
                            # Initialise 'start_i_match' which store starting character index.
                            if matches == 0:
                                start_i_match = start_i

                            # Track matches.
                            if (word_match == word_text_lower[:len(word_match)]) or (word_match == self.format_string(string=word_text_lower)):
                                matches += 1
                            else:
                                matches = 0

                            # Assert indexing correct.
                            assert word_text == text[start_i:end_i], \
                                "word_text: '{}' , text[start_i:end_i]: '{}'".format(
                                    word_text, text[start_i:end_i])

                            # Assert indexing correct (lower).
                            assert word_text_lower == text_lower[start_i_lower:end_i_lower], \
                                "word_text.lower(): '{}' , text_lower[start_i:end_i]: '{}'".format(
                                    word_text_lower, text_lower[start_i_lower:end_i_lower])

                            # Process entity link if all word stems are matches.
                            if matches == required_matches:

                                end_i_match = end_i
                                # Find true end token (i.e. remove puntuation at end)
                                while text[end_i_match - 1].isalpha() == False:
                                    if end_i_match <= start_i_match:
                                        break
                                    end_i_match += -1
                                # Fine true start token (i.e. remove puntuation at start)
                                while text[start_i_match].isalpha() == False:
                                    if end_i_match <= start_i_match:
                                        break
                                    start_i_match += 1

                                if end_i_match > start_i_match:
                                    # Create new EntityLink.AnchorTextLocation message for EntityLink.anchor_text_location.
                                    anchor_text_location = EntityLink.AnchorTextLocation()
                                    anchor_text_location.start = start_i_match
                                    anchor_text_location.end = end_i_match

                                    # Create new EntityLink message.
                                    synthetic_entity_link = EntityLink()
                                    synthetic_entity_link.anchor_text = text[start_i_match:end_i_match]
                                    synthetic_entity_link.entity_id = entity_id
                                    synthetic_entity_link.entity_name = entity_ids_to_entity_names[entity_id]
                                    synthetic_entity_link.anchor_text_location.MergeFrom(anchor_text_location)

                                    document_content.synthetic_entity_links.append(synthetic_entity_link)
                                matches = 0

                            # Add one character for space.
                            end_i += 1
                            end_i_lower += 1
                            # Shift word.
                            start_i = end_i
                            start_i_lower = end_i_lower

            
    def __get_entity_link_totals_from_document_contents(self, document_contents, link_type='MANUAL'):
        """ Append entity_link_totals features (protocol_buffers/document.proto:EntityLinkTotals) from
        document_contents. """
        assert link_type == 'MANUAL' or 'SYNTHETIC', "link_type: {} not 'MANUAL' or 'SYNTHETIC' flag".format(link_type)

        def get_correct_entity_links(document_content, link_type):
            """ Return list of EntityLink message for either 'MANUAL' or 'SYNTHETIC' links. """
            if link_type == 'MANUAL':
                return document_content.manual_entity_links
            else:
                return document_content.synthetic_entity_links

        # Build set of unique 'entity_id's of entities linked in document_contents
        entity_ids = []
        for document_content in document_contents:
            entity_links = get_correct_entity_links(document_content=document_content, link_type=link_type)

            for entity_link in entity_links:
                entity_ids.append(entity_link.entity_id)
        unique_entity_ids = set(entity_ids)

        entity_link_totals = []
        for unique_entity_id in unique_entity_ids:
            # For each unique 'entity_id' loop over 'document_contents' to build anchor_text_data.
            # 'anchor_text_data': key= 'anchor_text' of 'entity_id', value= number of times 'anchor_text' linked to
            # 'entity_id' i.. frequency.
            anchor_text_data = {}
            for document_content in document_contents:
                entity_links = get_correct_entity_links(document_content=document_content, link_type=link_type)

                for entity_link in entity_links:
                    if entity_link.entity_id == unique_entity_id:
                        entity_name = entity_link.entity_name
                        anchor_text = entity_link.anchor_text

                        if anchor_text not in anchor_text_data:
                            anchor_text_data[anchor_text] = 1
                        else:
                            anchor_text_data[anchor_text] += 1

            # For each unique 'entity_id' transform 'anchor_text_data' to required data format for 'anchor_text_frequency'
            anchor_text_frequencies = []
            for k, v in anchor_text_data.items():
                anchor_text_frequency = EntityLinkTotal.AnchorTextFrequency()
                anchor_text_frequency.anchor_text = k
                anchor_text_frequency.frequency = v
                anchor_text_frequencies.append(anchor_text_frequency)

            # Create an EntityLinkTotal message for 'entity_id'.
            entity_link_total = EntityLinkTotal()
            entity_link_total.entity_id = unique_entity_id
            entity_link_total.entity_name = entity_name
            entity_link_total.anchor_text_frequencies.extend(anchor_text_frequencies)

            # Append to Document.entity_link_totals
            entity_link_totals.append(entity_link_total)

        return entity_link_totals


    def __add_manual_and_sythetic_entity_link_totals(self):
        """ Add manual and synthetic entity link totals to document. """
        # Add list of manually tagged EntityLinkTotal messages by parsing list of DocumentContents messages.
        manual_entity_link_totals = self.__get_entity_link_totals_from_document_contents(
            document_contents=self.document.document_contents, link_type='MANUAL')
        self.document.manual_entity_link_totals.extend(manual_entity_link_totals)

        # Add list of synthetically tagged EntityLinkTotal messages by parsing list of DocumentContents messages.
        synthetic_entity_link_totals = self.__get_entity_link_totals_from_document_contents(
            document_contents=self.document.document_contents, link_type='SYNTHETIC')
        self.document.synthetic_entity_link_totals.extend(synthetic_entity_link_totals)


    def write_documents_to_file(self, path, documents, buffer_size=10):
        """ Write list of Documents messages to binary file. """
        stream.dump(path, *documents, buffer_size=buffer_size)


    def get_list_protobuf_messages(self, path):
        """ Retrieve list of protocol buffer messages from binary fire """
        return [d for d in stream.parse(path, Document)]


    def get_protobuf_message(self, path, doc_id):
        """ Retrieve protocol buffer message matching 'doc_id' from binary fire """
        return [d for d in stream.parse(path, Document) if d.doc_id == doc_id][0]


    def __add_document_metadata(self, page):
        """ Add metadata to Document message. """
        # Unique id of document.
        self.document.doc_id = page.page_id
        # Title of document.
        self.document.doc_name = page.page_name
        # Page type: { “ArticlePage”,  “CategoryPage”, “RedirectPage “ + “ParaLink”, “DisambiguationPage” }
        self.document.page_type = str(page.page_type)
        # List of other documents that redirect to this document.
        self.redirect_names = page.page_meta.redirectNames
        # List of Wikipedia document ids within DisambiguationPage for this document.
        self.disambiguation_ids = page.page_meta.disambiguationIds
        # List of Wikipedia document names within DisambiguationPage for this document.
        self.disambiguation_names = page.page_meta.disambiguationNames
        # List of Wikipedia CategoryPages ids this document belongs.
        self.category_ids = page.page_meta.categoryIds
        # List of Wikipedia CategoryPages names this document belongs.
        self.category_names = page.page_meta.categoryNames
        # List of document ids that manually link to this document.
        self.manual_inlink_ids = page.page_meta.inlinkIds
        # List of ManualInlinkAnchors (Anchor text, frequency) that manually link to this document.
        self.manual_inlink_anchors = page.page_meta.inlinkAnchors


    def parse_page_to_protobuf(self, page):
        """ Parse trec-car-tools Page object to create Document message with synthetic entity linking. """
        # Initialise empty message.
        self.document = Document()

        # Add metadata to message.
        self.__add_document_metadata(page=page)

        # Add list of DocumentContents messages by parsing Page.skeleton (including manual linking).
        self.__add_document_contents_from_page_skeleton(skeleton=page.skeleton)

        # Add synthetic tags to all DocumentContents.
        self.__add_synthetic_entity_links()

        # Add list of manually and synthetic tagged EntityLinkTotal messages by parsing list of
        # DocumentContents messages.
        self.__add_manual_and_sythetic_entity_link_totals()

        return self.document


    def parse_cbor_to_protobuf(self, read_path, write_path, num_docs, buffer_size=10, print_intervals=100,
                               write_output=True):
        """ Read TREC CAR cbor file to create a list of protobuffer Document messages
        (protocol_buffers/document.proto:Documents).  This list of messages are streammed to binary file using 'stream'
        package. """
        # list of Document messages.
        documents = []
        t_start = time.time()

        with open(read_path, 'rb') as f_read:

            # Loop over Page objects
            for i, page in enumerate(iter_pages(f_read)):

                # Stops when 'num_pages' have been processed.
                if i >= num_docs:
                    break

                # parse page to create new document.
                self.parse_page_to_protobuf(page=page)

                # Append Document message to document list.
                documents.append(self.document)

                # Prints updates at 'print_pages' intervals.
                if ((i+1) % print_intervals == 0):
                    print('----- DOC #{} -----'.format(i))
                    print(self.document.doc_id)
                    time_delta = time.time() - t_start
                    print('time elapse: {} --> time / page: {}'.format(time_delta, time_delta / (i + 1)))

        if write_output:
            print('STEAMING DATA TO FILE: {}'.format(write_path))
            self.write_documents_to_file(path=write_path, documents=documents, buffer_size=buffer_size)
            print('FILE WRITTEN')

        time_delta = time.time() - t_start
        print('PROCESSED DATA: {} --> processing time / page: {}'.format(time_delta, time_delta / (i + 1)))


if __name__ == '__main__':
    # name = 'custom'
    # year = 1
    # read_path = '/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark_processing/data/testY{}.pages.cbor'.format(year)
    # write_path = '/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark_processing/data/testY{}_{}.bin'.format(year, name)

    for i in range(0,5):
        read_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/fold-{}-train.pages.cbor'.format(i)
        write_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/fold-{}-train.pages.bin'.format(i)
        num_docs = 10000
        write_output = True
        print_intervals = 10
        buffer_size = 10
        parser = TrecCarParser()
        parser.parse_cbor_to_protobuf(read_path=read_path, write_path=write_path, num_docs=num_docs,
                                      buffer_size=buffer_size, print_intervals=print_intervals, write_output=write_output)
