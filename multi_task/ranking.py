
from scipy.spatial import distance
import json
import os


def get_dict_from_json(path):
    """"""
    with open(path, 'r') as f:
        d = json.load(f)
    return d

def write_run_to_file(query, run_data, path, how):
    """ """
    run_data.sort(key=lambda tup: tup[1], reverse=True)
    rank = 1
    with open(path, 'a+') as f:
        for doc_id, score in run_data:
            f.write(' '.join(query, 'Q0', doc_id, rank, score, how) + '\n')
            rank += 1

            
def get_read_data(dataset, how='euclidean', parent_dir_path='/nfs/trec_car/data/entity_ranking/multi_task_data_by_query/'):
    """ """
    dir_path = parent_dir_path + '{}_data/'.format(dataset)

    entity_links_path = dir_path + 'passage_to_entity.json'
    entity_links_dict = get_dict_from_json(path=entity_links_path)

    for query_path in [dir_path + f for f in os.listdir(dir_path) if 'data.json' in f]:
        query_dict = get_dict_from_json(path=query_path)
        query = query_dict['query']['query_id']

        query_cls = query_dict['query']['cls_token']

        run = []
        for doc_id in query_dict['passage'].keys():
            doc_cls = query_dict['passage'][doc_id]['cls_token']

            if how == 'euclidean' :
                score = - distance.euclidean(query_cls,  doc_cls)
                path = dir_path + how + '_passage.run'
            else:
                raise

            run.append((doc_id, score))

        write_run_to_file(query=query, run=run, path=path, how=how)
