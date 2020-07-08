
from document_parsing.trec_car_parsing import TrecCarParser

import numpy as np
import urllib
import re

def check_manual_vs_synthetic_links():
    """ Util to check manual vs. synthetic entity linking. """
    path = '/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark/data/data_proto_Y2_custom_end_punt_v6.bin'
    tcp = TrecCarParser()

    document_list = tcp.get_list_protobuf_messages(path=path)
    for doc in document_list:
        if 'enwiki:Blue-ringed%20octopus' in doc.doc_id:

            for document_content in doc.document_contents:
                manual_link_list = []
                manual_entity_id_data ={}
                for manual_link in document_content.manual_entity_links:
                    manual_link_list.append(manual_link.entity_id)
                    if manual_link.entity_id not in manual_entity_id_data:
                        manual_entity_id_data[manual_link.entity_id] = {
                            'entity_name': manual_link.entity_name,
                            'anchor_text': [(manual_link.anchor_text, manual_link.anchor_text_location)]
                        }
                    else:
                        manual_entity_id_data[manual_link.entity_id]['anchor_text'].append((manual_link.anchor_text, manual_link.anchor_text_location))
                synthetic_link_list = []
                for synthetic_link in document_content.synthetic_entity_links:
                    synthetic_link_list.append(synthetic_link.entity_id)

                missed_manual_entity_ids = list(np.setdiff1d(manual_link_list, synthetic_link_list))
                if len(missed_manual_entity_ids) > 0:
                    print('===================')
                    print(document_content.content_id)
                    print(doc.doc_id, doc.doc_name, document_content.section_heading_names)
                    print('manual: {}'.format(sorted(manual_link_list)))
                    print('synthetic: {}'.format(sorted(synthetic_link_list)))
                    print('***')
                    print(document_content.text)
                    for entity_id in missed_manual_entity_ids:
                        print(entity_id, manual_entity_id_data[entity_id]['entity_name'])
                        for i in manual_entity_id_data[entity_id]['anchor_text']:
                            print(i[0], i[1], document_content.text[i[1].start:i[1].end])


def get_query(doc_id, section_heading_names):
    """ Build query with doc_id and section_heading_names. """
    if len(section_heading_names) > 0:
        encoded_section_heading_names = [urllib.parse.quote(string=text, encoding='utf-8') for text in section_heading_names]
        terms = [doc_id] + encoded_section_heading_names
        return '/'.join(terms)
    else:
        return doc_id


def build_synthetic_qrels(document_list, path, qrels_type='tree', write=True):
    """ Build tree or hierarchical qrels. """
    # Build hierarchical qrels.
    hierarchical_qrels = {}
    # Store list of section_heading_names for each doc_id
    doc_id_section_heading_names_list = {}
    for doc in document_list:
        doc_id_section_heading_names_list[doc.doc_id] = []
        for document_content in doc.document_contents:

            # Build query.
            query = get_query(doc_id=doc.doc_id, section_heading_names=document_content.section_heading_names)

            if len(document_content.section_heading_names) > 0:
                doc_id_section_heading_names_list[doc.doc_id].append(document_content.section_heading_names)

            # Populate hierarchical qrels.
            if query not in hierarchical_qrels:
                hierarchical_qrels[query] = []
            # Add synthetic links.
            for synthetic_link in document_content.synthetic_entity_links:
                if synthetic_link.entity_id not in hierarchical_qrels[query]:
                    hierarchical_qrels[query].append(synthetic_link.entity_id)
            # Add manual links.
            for manual_link in document_content.manual_entity_links:
                if manual_link.entity_id not in hierarchical_qrels[query]:
                    hierarchical_qrels[query].append(manual_link.entity_id)

    if qrels_type == 'tree':
        # Build list of tree queries for each doc_id.
        doc_id_tree_queries = {}
        for doc_id, section_heading_names in doc_id_section_heading_names_list.items():
            # Build query.
            query = get_query(doc_id=doc_id, section_heading_names=[])
            doc_id_tree_queries[doc_id] = [query]
            unique_section_heading_names = sorted([list(headers) for headers in set(tuple(headers) for headers in section_heading_names)])
            for headers in unique_section_heading_names:
                partial_h = []
                for h in headers:
                    partial_h.append(h)
                    query = get_query(doc_id=doc_id, section_heading_names=partial_h)
                    if query not in doc_id_tree_queries[doc_id]:
                        doc_id_tree_queries[doc_id].append(query)

        # Build tree qrels.
        tree_qrels = {}
        for doc_id, tree_queries in doc_id_tree_queries.items():
            for tree_query in tree_queries:
                for hierarchical_query in hierarchical_qrels.keys():
                    if (tree_query == hierarchical_query[:len(tree_query)]):
                        if tree_query in tree_qrels:
                            doc_ids = list(set(tree_qrels[tree_query] + hierarchical_qrels[hierarchical_query]))
                            tree_qrels[tree_query] = doc_ids
                        else:
                            tree_qrels[tree_query] = hierarchical_qrels[hierarchical_query]

        qrels = tree_qrels

    elif qrels_type == 'top-level':
        qrels = {}
        for k, v in hierarchical_qrels.items():
            if k.count('/') >= 2:
                i = [m.start() for m in re.finditer(r"/", k)][1]
                new_k = k[:i]
                if new_k in qrels:
                    qrels[new_k] = list(set(v + qrels[new_k]))
                else:
                    qrels[new_k] = v
            else:
                if k in qrels:
                    qrels[k] = list(set(v + qrels[new_k]))
                else:
                    qrels[k] = v

    elif qrels_type == 'hierarchical':
        qrels = hierarchical_qrels

    else:
        "Not valid qrels_type: {}".qrels_type
        raise

    # Write to file
    if write:
        with open(path, 'w') as f:
            for query in sorted(qrels.keys()):
                for doc in sorted(qrels[query]):
                    f.write('{} 0 {} 1\n'.format(query, doc))


def build_passage_qrels(document_list, path):
    """ Build hierarchical passage qrels. """
    # Build hierarchical qrels.
    qrels = {}
    # Store list of section_heading_names for each doc_id
    doc_id_section_heading_names_list = {}
    for doc in document_list:
        doc_id_section_heading_names_list[doc.doc_id] = []
        for document_content in doc.document_contents:

            # Build query.
            query = get_query(doc_id=doc.doc_id, section_heading_names=document_content.section_heading_names)

            if len(document_content.section_heading_names) > 0:
                doc_id_section_heading_names_list[doc.doc_id].append(document_content.section_heading_names)

            # Populate hierarchical qrels.
            if query not in qrels:
                qrels[query] = []
            # Add synthetic links.
            if len(document_content.content_id) > 0:
                qrels[query].append(document_content.content_id)

    # Write to file
    with open(path, 'w') as f:
        for query in sorted(qrels.keys()):
            for passage in sorted(qrels[query]):
                f.write('{} 0 {} 1\n'.format(query, passage))


if __name__ == '__main__':
    import pandas as pd

    proto_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/fold-0-train.pages.bin'
    tcp = TrecCarParser()
    document_list = tcp.get_list_protobuf_messages(path=proto_path)
    path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/benchmarkY1.dev.top-level.synthetic.qrels'
    build_synthetic_qrels(document_list, path, qrels_type='top-level', write=True)

    document_list = []
    for i in range(1,5):
        proto_path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/fold-{}-train.pages.bin'.format(i)
        tcp = TrecCarParser()
        document_list += tcp.get_list_protobuf_messages(path=proto_path)

    path = '/Users/iain/LocalStorage/coding/github/multi-task-ranking/data/temp/benchmarkY1.train.top-level.synthetic.qrel'
    build_synthetic_qrels(document_list=document_list, path=path, qrels_type='top-level', write=True)

    #build_passage_qrels(document_list=document_list, path=path)
    # data = []
    # for doc in document_list:
    #     doc_id = doc.doc_id
    #     for document_content in doc.document_contents:
    #         if len(document_content.content_id) > 0:
    #             content_id = document_content.content_id
    #             section_heading_ids = document_content.section_heading_ids
    #             text = document_content.text
    #             id_in_Y2_goldpassages_qrels = content_id in valid_qrels
    #             data.append([content_id, doc_id, text, section_heading_ids, id_in_Y2_goldpassages_qrels])
    #
    # import pandas as pd
    #
    # df = pd.DataFrame(data, columns=['id', 'doc_id', 'text', 'section_heading_ids','id_in_Y2_goldpassages_qrels'])
    # df.to_csv('/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark_processing/data/testY2.v4.csv')

    #print(document_list[0])
    # doc = tcp.get_protobuf_message(path=proto_path, doc_id=doc_id)
    # print(doc)
    #build_synthetic_qrels(document_list=document_list, path=path, qrels_type=qrels_type)



