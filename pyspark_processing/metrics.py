
from document_parsing.trec_car_parsing import TrecCarParser

import os
#TODO - token precion and recall

def get_true_and_false_linking_counts(document, ground_truth, metric, link_type):
    """ Sums Positives (P) and False Negatives (FN) or False Positives (FP) from Document given ground truth Document.
    Results are returned from 'exact_match' (i.e. EntityLink identical) and 'fuzzy_match' (i.e. EntityLinks that have
    anchor text spans that cross and 'entity_id' match). """

    def get_entity_links_from_doc_content(doc_content, link_type):
        """ Get manual or synthetic entity links. """
        if link_type == 'manual':
            return doc_content.manual_entity_links
        elif link_type == 'synthetic':
            return doc_content.synthetic_entity_links
        else:
            "ERROR"
            raise

    def exact_match(doc_outer_entity_link, doc_inner_content, link_type):
        """ Return True if doc_outer_entity_link has an identical EntityLink in doc_inner_content. """
        # For all EntityLinks in doc_inner_content.
        doc_inner_entity_links = get_entity_links_from_doc_content(doc_content=doc_inner_content, link_type=link_type)

        for doc_inner_entity_link in doc_inner_entity_links:
            if doc_outer_entity_link == doc_inner_entity_link:
                # Found exact match.
                return True

        # Cannot find exact match, returning False.
        return False


    def fuzzy_match(doc_outer_entity_link, doc_inner_content, link_type):
        """ Return True if doc_outer_entity_link has an fuzzy EntityLink match doc_inner_content. This is defined as
        having anchor text spans that cross and 'entity_id' match """

        # Unpack doc_outer_entity_link data.
        outer_start = doc_outer_entity_link.anchor_text_location.start
        outer_end = doc_outer_entity_link.anchor_text_location.end
        outer_entity_id = doc_outer_entity_link.entity_id

        # For all EntityLinks in doc_inner_content.
        doc_inner_entity_links = get_entity_links_from_doc_content(doc_content=doc_inner_content, link_type=link_type)
        for doc_inner_entity_link in doc_inner_entity_links:
            # Unpack doc_inner_entity_link data.
            inner_entity_id = doc_inner_entity_link.entity_id
            inner_start = doc_inner_entity_link.anchor_text_location.start
            inner_end = doc_inner_entity_link.anchor_text_location.end

            # If anchor text spans cross and 'entity_id' are the same.
            if outer_entity_id == inner_entity_id:
                for i in range(outer_start, outer_end+1):
                    if (inner_start <= i <= inner_end):
                        # Found fuzzy match.
                        return True

        # Cannot find fuzzy match, returning False.
        return False


    # Asserts user has set valid metric.
    assert metric == 'FN' or metric == 'FP', "NEED TO SET 'metric' flag to 'FN' or 'FP' "
    assert link_type == 'manual' or link_type == 'synthetic',  "NEED TO SET 'link_type' flag to 'manual' or 'synthetic' "

    # Given metric specified, set 'ground_truth' and 'document' to 'doc_outer' and 'doc_inner'.
    if metric == 'FN':
        doc_outer = ground_truth
        doc_inner = document
        doc_outer_link_type = 'synthetic'
        doc_inner_link_type = link_type
    elif metric == 'FP':
        doc_outer = document
        doc_inner = ground_truth
        doc_inner_link_type = 'synthetic'
        doc_outer_link_type = link_type
    else:
        print('NEED VALID METRIC REQUEST')
        raise

    # Counters for True (T), True Fuzzy (T_fuzzy), False (F), False Fuzzy (F_fuzzy)
    T, T_fuzzy, F, F_fuzzy = 0, 0, 0, 0
    results_dict = {}

    index = 0
    for doc_outer_content, doc_inner_content in zip(doc_outer.document_contents, doc_inner.document_contents):

        doc_outer_entity_links = get_entity_links_from_doc_content(doc_content=doc_outer_content, link_type=doc_outer_link_type)

        for doc_outer_entity_link in doc_outer_entity_links:
            if doc_outer_entity_link.entity_id not in results_dict:
                results_dict[doc_outer_entity_link.entity_id] = {
                    'T': 0,
                    'T_fuzzy': 0,
                    'F': 0,
                    'F_fuzzy': 0
                }

            # Exact match.
            if exact_match(doc_outer_entity_link=doc_outer_entity_link,
                           doc_inner_content=doc_inner_content,
                           link_type=doc_inner_link_type):
                T += 1
                T_fuzzy += 1
                results_dict[doc_outer_entity_link.entity_id]['T'] += 1
                results_dict[doc_outer_entity_link.entity_id]['T_fuzzy'] += 1

            # Fuzzy match.
            elif fuzzy_match(doc_outer_entity_link=doc_outer_entity_link,
                             doc_inner_content=doc_inner_content,
                             link_type=doc_inner_link_type):
                T_fuzzy += 1
                F += 1
                results_dict[doc_outer_entity_link.entity_id]['F'] += 1
                results_dict[doc_outer_entity_link.entity_id]['T_fuzzy'] += 1

            # No match.
            else:
                F += 1
                F_fuzzy += 1
                results_dict[doc_outer_entity_link.entity_id]['F'] += 1
                results_dict[doc_outer_entity_link.entity_id]['F_fuzzy'] += 1

        index += 1

    return T, F, T_fuzzy, F_fuzzy, results_dict


def get_recall(T, FN):
    """ Calculates recall score. """
    if (T + FN) > 0:
        return T / (T + FN)
    return 1.0


def get_precision(T, FP):
    """ Calculates precision score. """
    if (T + FP) > 0:
        return T / (T + FP)
    return 1.0


def get_F1(precision, recall):
    """ Calculates F1 score. """
    if (precision + recall) > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 1.0


def combine_results_dicts(d_FN, d_FP):
    """ Print stats for all entity links. """
    for i in d_FP.keys():
        T1 = d_FP[i]['T']
        FP = d_FP[i]['T_fuzzy']
        T1_fuzzy = d_FP[i]['F']
        FP_fuzzy = d_FP[i]['F_fuzzy']

        T2 = d_FN[i]['T']
        FN = d_FN[i]['T_fuzzy']
        T2_fuzzy = d_FN[i]['F']
        FN_fuzzy = d_FN[i]['F_fuzzy']

        print('{} ======'.format(i))
        print_func(T1, FP, T1_fuzzy, FP_fuzzy, T2, FN, T2_fuzzy, FN_fuzzy)


def print_func(T1, FP, T1_fuzzy, FP_fuzzy, T2, FN, T2_fuzzy, FN_fuzzy):
    """ Print function for page. """
    #T = (T1 + T2) / 2
    T = T2
    #T_fuzzy = (T1_fuzzy + T2_fuzzy) / 2
    T_fuzzy = T2_fuzzy
    # Assert both calculated P are equal.
    #assert T1 == T2
    # assert T1_fuzzy == T2_fuzzy, "T1_fuzzy {} vs. T2_fuzzy {}".format(T1_fuzzy, T2_fuzzy)

    # Calculate recall, precision and F1.
    recall = get_recall(FN=FN, T=T)
    precision = get_precision(FP=FP, T=T)
    F1 = get_F1(precision=precision, recall=recall)

    # Calculate recall, precision and F1.
    recall_fuzzy = get_recall(FN=FN_fuzzy, T=T_fuzzy)
    precision_fuzzy = get_precision(FP=FP_fuzzy, T=T_fuzzy)
    F1_fuzzy = get_F1(precision=precision_fuzzy, recall=recall_fuzzy)

    # Print metrics.
    print('*EXACT*')
    print("Precision\tRecall\tF1\tTP\tFP\tFN")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}".format(precision, recall, F1, T, FP, FN))
    print('*FUZZY*')
    print("Precision\tRecall\tF1\tTP\tFP\tFN")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}".format(precision_fuzzy, recall_fuzzy, F1_fuzzy, T_fuzzy, FP_fuzzy, FN_fuzzy))

    return T1, FP, T1_fuzzy, FP_fuzzy

def print_stats(document, ground_truth, doc_id, link_type, print_links=False):
    """ Prints metrics (precision, recall, F1) """
    # Calculate False Positives (FP) and Positives (P).
    T1, FP, T1_fuzzy, FP_fuzzy, results_dict_FP = get_true_and_false_linking_counts(
        document=document, ground_truth=ground_truth, metric='FP', link_type=link_type)

    # Calculate False Negatives (FN) and Positives (P).
    T2, FN, T2_fuzzy, FN_fuzzy, results_dict_FN = get_true_and_false_linking_counts(
        document=document, ground_truth=ground_truth, metric='FN', link_type=link_type)

    if print_links:
        combine_results_dicts(d_FN=results_dict_FN, d_FP=results_dict_FP)

    print('=== {} {} ==='.format(doc_id, link_type))
    return print_func(T1, FP, T1_fuzzy, FP_fuzzy, T2, FN, T2_fuzzy, FN_fuzzy)


def calculate_document_metrics(annotated_metadata, print_links):
    """ Calculate metrics (precision, recall, F1) for 'doc_id' Document given ground truth Document. """
    # Initialise TrecCarParser.
    parser = TrecCarParser()

    for ground_truth_path, document_path, doc_id in annotated_metadata:

        # Read ground truth and document to be evaluated.
        document = parser.get_protobuf_message(path=document_path, doc_id=doc_id)
        ground_truth = parser.get_protobuf_message(path=ground_truth_path, doc_id=doc_id)

        print('\n  ///// {} /////'.format(doc_id))
        # Print stats of document given ground_truth.
        #print_stats(document=document, ground_truth=ground_truth, doc_id=doc_id, link_type='manual', print_links=print_links)

        # Print stats of document given ground_truth.
        print_stats(document=document, ground_truth=ground_truth, doc_id=doc_id, link_type='synthetic', print_links=print_links)

if __name__ == "__main__":

    base_path = '/Users/iain/LocalStorage/coding/github/entity-linking-with-pyspark/data/'
    suffix = 'regex'
    annotated_metadata = [
        (
            base_path + 'data_proto_ground_truth_Aftertaste.bin',
            base_path + 'testY1_{}.bin'.format(suffix),
            'enwiki:Aftertaste'
        ),
        (
            base_path + 'data_proto_ground_truth_Blue-ringed%20octopus.bin',
            base_path + 'testY2_{}.bin'.format(suffix),
            'enwiki:Blue-ringed%20octopus'
        ),
        (
            base_path + 'data_proto_ground_truth_Aerobic%20fermentation.bin',
            base_path + 'testY2_{}.bin'.format(suffix),
            'enwiki:Aerobic%20fermentation'
        ),
    ]
    print_links = False
    calculate_document_metrics(annotated_metadata=annotated_metadata, print_links=print_links)
