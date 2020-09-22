
from scipy.spatial import distance
import json
import os

from multi_task.processing import dataset_metadata
from retrieval.tools import EvalTools


def get_dict_from_json(path):
    """"""
    with open(path, 'r') as f:
        d = json.load(f)
    return d


def write_run_to_file(query, run_data, run_path, how):
    """ """
    run_data.sort(key=lambda tup: tup[1], reverse=True)
    rank = 1
    with open(run_path, 'a+') as f:
        for doc_id, score in run_data:
            f.write(' '.join((query, 'Q0', doc_id, str(rank), str(score), how)) + '\n')
            rank += 1


def rerank_runs(dataset, how='euclidean', parent_dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'):
    """ """
    dir_path = parent_dir_path + '{}_data/'.format(dataset)

    passage_qrels = dataset_metadata['passage_' + dataset][1]
    entity_qrels = dataset_metadata['entity_' + dataset][1]


    entity_links_path = dir_path + 'passage_to_entity.json'
    entity_links_dict = get_dict_from_json(path=entity_links_path)

    for query_path in [dir_path + f for f in os.listdir(dir_path) if 'data.json' in f]:
        query_dict = get_dict_from_json(path=query_path)
        query = query_dict['query']['query_id']

        query_cls = query_dict['query']['cls_token']

        run_data = []
        for doc_id in query_dict['passage'].keys():
            doc_cls = query_dict['passage'][doc_id]['cls_token']

            if how == 'euclidean' :
                score = - distance.euclidean(query_cls,  doc_cls)
                run_path = dir_path + how + '_passage.run'
            else:
                raise

            run_data.append((doc_id, score))

        write_run_to_file(query=query, run_data=run_data, run_path=run_path, how=how)
        EvalTools().write_eval_from_qrels_and_run(qrels_path=passage_qrels, run_path=run_path)