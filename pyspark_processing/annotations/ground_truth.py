
from document_parsing.trec_car_parsing import TrecCarParser
from protocol_buffers.document_pb2 import Document, DocumentContent, EntityLink


def add_extra_annotations_to_document(read_path, write_path, doc_id, annotations):
    """ Augment a Document message with ground truth linking annotations. """
    # Initialise TrecCarParser object and read binary file from 'path' and returning 'doc_id' Document message.
    parser = TrecCarParser()
    document = parser.get_protobuf_message(path=read_path, doc_id=doc_id)

    # Start new Document message.
    new_document = Document()
    # Add metadata (doc_id & doc_name).
    new_document.doc_id = document.doc_id
    new_document.doc_name = document.doc_name

    # Build list of valid entity ids i.e. original entity links added by Wikipedia.
    manual_entity_ids = []
    for document_content in document.document_contents:
        for manual_entity_link in document_content.manual_entity_links:
            manual_entity_ids.append(manual_entity_link.entity_id)
    valid_entity_ids = list(set(manual_entity_ids))

    # Loop over all DocumentContents messages to augment entity links.
    for i, document_content in enumerate(document.document_contents):

        # Start new DocumentContent message.
        new_document_content = DocumentContent()

        # Set fields of new DocumentContent to original values.
        new_document_content.content_id = document_content.content_id
        new_document_content.section_heading_ids.extend(document_content.section_heading_ids)
        new_document_content.section_heading_names.extend(document_content.section_heading_names)
        new_document_content.content_type = document_content.content_type
        new_document_content.text = document_content.text
        new_document_content.content_urls.extend(document_content.content_urls)
        new_document_content.manual_entity_links.extend(document_content.manual_entity_links)

        # If annotations in DocumentContent augment annotations.
        if len(annotations) > i:
            print('BUILDING ANNOTATIONS FOR INDEX #{}'.format(i))
            # Loop over annotations list.
            for a in annotations[i]:

                # Get text span of annotated anchor text.
                text_span = document_content.text[a.anchor_text_location.start:a.anchor_text_location.end]

                # Assert anchor text matches text span.
                assert a.anchor_text == text_span, "Index: {}, Character: {} anchor_test: {} v.s. character index: {}"\
                    .format(i, a.anchor_text_location.start, a.anchor_text, text_span)
                # Assert valid entity link.
                assert a.entity_id in valid_entity_ids, "NOT VALID ENTITY LINKS: {} not in {}".format(
                    a.entity_id , valid_entity_ids)

                # Construct EntityLink.AnchorTextLocation message.
                anchor_text_location = EntityLink.AnchorTextLocation()
                anchor_text_location.start = a.anchor_text_location.start
                anchor_text_location.end = a.anchor_text_location.end

                # Construct EntityLink message
                entity_link = EntityLink()
                entity_link.anchor_text = a.anchor_text
                entity_link.anchor_text_location.MergeFrom(anchor_text_location)
                entity_link.entity_id = a.entity_id
                entity_link.entity_name = a.entity_name

                # Append EntityLink message to DocumentContents
                new_document_content.synthetic_entity_links.append(entity_link)

        else:
            print('NO ANNOTATIONS FOR INDEX #{}'.format(i))

        # Append DocumentContents with new entity links to Document.document_contents.
        new_document.document_contents.append(new_document_content)

    # # Append DocumentContents with new entity links to Document.document_contents.
    # manual_entity_link_totals = parser.get_entity_link_totals_from_document_contents(
    #     document_contents=new_document.document_contents, link_type='MANUAL')
    # new_document.manual_entity_link_totals.extend(manual_entity_link_totals)
    #
    # synthetic_entity_link_totals = parser.get_entity_link_totals_from_document_contents(
    #     document_contents=new_document.document_contents, link_type='SYNTHETIC')
    # new_document.synthetic_entity_link_totals.extend(synthetic_entity_link_totals)

    # Write new Document to binary file.
    parser.write_documents_to_file(path=write_path, documents=[new_document], buffer_size=1)


if __name__ == '__main__':
    read_path = '/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark/data/testY2_refactor.bin'
    write_path = '/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark/data/data_proto_ground_truth_Blue-ringed%20octopus.bin'
    doc_id = 'enwiki:Blue-ringed%20octopus'
    from pyspark_processing.annotations.annotation_data import annotation_dict

    new_document = add_extra_annotations_to_document(read_path=read_path,
                                                     write_path=write_path,
                                                     doc_id=doc_id,
                                                     annotations=annotation_dict[doc_id])
