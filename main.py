
import os

from metadata import CarEntityPaths, CarPassagePaths, NewsPassagePaths
from learning.experiments import FineTuningReRankingExperiments
from retrieval.dataset_processing import DatasetProcessing, BertTokenizer
from retrieval.tools import EvalTools, SearchTools, default_eval_config
from torch import nn

if __name__ == '__main__':

    # from metadata import NewsPassagePaths, CarEntityPaths
    # from retrieval.tools import SearchTools

    # run_bases = ['/nfs/trec_news_track/runs/anserini/entity_2018/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run',
    #              '/nfs/trec_news_track/runs/anserini/entity_2019/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run',
    #              '/nfs/trec_news_track/runs/anserini/entity_2018/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run',
    #              '/nfs/trec_news_track/runs/anserini/entity_2019/entity.custom_anserini.500000_doc.100_words.{}.fixed_qrels.run']
    #
    # qrels_paths = ['/nfs/trec_news_track/data/2018/news_track.2018.entity.qrels',
    #                '/nfs/trec_news_track/data/2019/news_track.2019.entity.qrels',
    #                '/nfs/trec_news_track/data/2018/news_track.2018.entity.qrels',
    #                '/nfs/trec_news_track/data/2019/news_track.2019.entity.qrels'
    #                ]
    #
    # query_types = ['title+contents',
    #                'title+contents',
    #                'title',
    #                'title']

    # run_paths = ['/nfs/trec_news_track/runs/anserini/bert/news_track.dev.bm25.100000.title+contents.50_words.run',
    #              '/nfs/trec_news_track/runs/anserini/bert/news_track.test.bm25.100000.title+contents.50_words.run']
    # qrels_paths = ['/nfs/trec_news_track/bert/dev_entity/news_track.dev.entity.qrels',
    #                '/nfs/trec_news_track/bert/test_entity/news_track.test.entity.qrels']
    # query_type = 'title+contents'
    #
    # hits = 100000
    # news_index_path = NewsPassagePaths.index
    # words = 50
    #
    # for run_path, qrels_path in zip(run_paths, qrels_paths):
    #     search_tools = SearchTools(index_path=CarEntityPaths.index)
    #     search_tools.write_entity_run_news(run_path, qrels_path, query_type, words, hits, news_index_path)

    # from retrieval.dataset_processing import DatasetProcessing
    # from metadata import NewsPassagePaths
    #
    #TEST
    # qrels_path = '/nfs/trec_news_track/bert/test_passage/news_track.test.passage.qrels'
    # xml_topics_path = '/nfs/trec_news_track/data/2019/newsir19-background-linking-topics.xml'
    # run_path = '/nfs/trec_news_track/bert/test_passage/news_track.test.passage.250.bm25.rm3.run'
    # index_path = NewsPassagePaths.index
    # car_index_path = None #CarEntityPaths.index
    # data_dir_path = '/nfs/trec_news_track/bert/test_passage/news_track_test_passage_250_bm25_rm3_bert_chunks_scaled_rel/'
    # max_length = 512
    # context_path = None
    # training_dataset = False
    # ranking_type = 'passage'
    # query_type = 'title+contents'
    #
    # processing = DatasetProcessing(qrels_path=qrels_path,
    #                                run_path=run_path,
    #                                index_path=index_path,
    #                                data_dir_path=data_dir_path,
    #                                max_length=max_length,
    #                                context_path=context_path)
    #
    # processing.build_news_dataset(training_dataset=training_dataset,
    #                               chuck_topic_size=1e8,
    #                               ranking_type=ranking_type,
    #                               query_type=query_type,
    #                               car_index_path=car_index_path,
    #                               xml_topics_path=xml_topics_path)

    gpus = 2
    model_path = None #'/nfs/trec_car/data/bert_reranker_datasets/exp/test_trec_news_v2_entity_4e5_batch_8_fixed_qrels_scaled_rel/epoch2_batch25/'
    extra_layers = False
    train_batch_size = 8 * gpus
    dev_batch_size = 64 * 3 * gpus

    train_data_dir_path_passage = '/nfs/trec_news_track/bert/train_passage/news_track_train_passage_250_bm25_rm3_bert_chunks_scaled_rel/'
    dev_data_dir_path_passage =  '/nfs/trec_news_track/bert/dev_passage/news_track_dev_passage_250_bm25_rm3_bert_chunks_scaled_rel/'
    dev_qrels_path_passage = '/nfs/trec_news_track/bert/dev_passage/news_track.dev.passage.qrels'
    dev_run_path_passage = '/nfs/trec_news_track/bert/dev_passage/news_track.dev.passage.250.bm25.rm3.run'

    # train_data_dir_path_entity = None #'/nfs/trec_news_track/runs/anserini/bert/news_track_train_bm25_100000_50_words_bert_chunks_scaled_rel/'
    # dev_data_dir_path_entity = '/nfs/trec_news_track/runs/anserini/bert/news_track_test_bm25_100000_50_words_bert_chunks_scaled_rel/'
    # dev_qrels_path_entity = '/nfs/trec_news_track/bert/test_entity/news_track.test.entity.qrels'
    # dev_run_path_entity = '/nfs/trec_news_track/runs/anserini/bert/news_track.test.bm25.100000.title+contents.50_words.run'

    experiment = FineTuningReRankingExperiments(model_path=model_path,
                                                extra_layers=extra_layers,
                                                train_batch_size=train_batch_size,
                                                dev_batch_size=dev_batch_size,

                                                # train_data_dir_path_entity=train_data_dir_path_entity,
                                                # dev_data_dir_path_entity=dev_data_dir_path_entity,
                                                # dev_qrels_path_entity=dev_qrels_path_entity,
                                                # dev_run_path_entity=dev_run_path_entity,

                                                train_data_dir_path_passage=train_data_dir_path_passage,
                                                dev_data_dir_path_passage=dev_data_dir_path_passage,
                                                dev_qrels_path_passage=dev_qrels_path_passage,
                                                dev_run_path_passage=dev_run_path_passage

                                                )

    epochs = 3
    lr = 2e-5
    eps = 1e-8
    weight_decay = 0.01
    warmup_percentage = 0.1
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'test_trec_news_v2_passage_2e5_batch_16_fixed_qrels_scaled_rel'
    write = True
    logging_steps = 100
    head_flag = 'passage'

    experiment.run_experiment_single_head(
        head_flag=head_flag,
        epochs=epochs,
        lr=lr,
        eps=eps,
        weight_decay=weight_decay,
        warmup_percentage=warmup_percentage,
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
        logging_steps=logging_steps
    )

    # head_flag = 'entity'
    # rerank_run_path = '/nfs/trec_news_track/runs/bert/entity_2018/test_entity_news_fixed_qrels_scaled_rel.bm5_re_rank.run'
    # experiment.inference(head_flag=head_flag, rerank_run_path=rerank_run_path, do_eval=False, cap_rank=100)
    #
    # from REL.mention_detection import MentionDetection
    # from REL.utils import process_results
    # from REL.entity_disambiguation import EntityDisambiguation
    # from REL.ner import Cmns, load_flair_ner
    # import time
    #
    # base_url = "/Users/iain/LocalStorage/coding/github/REL/"
    # wiki_year = '2014'
    # wiki_version = "wiki_" + wiki_year
    # model_path = "/Users/iain/LocalStorage/coding/github/REL/ed-wiki-{}/model".format(wiki_year)
    #
    #
    # def example_preprocessing():
    #     # user does some stuff, which results in the format below.
    #     text_1 = "Roger Federer (German pronunciation: [ˈrɔdʒər ˈfeːdərər]; born 8 August 1981) is a Swiss professional tennis player who is ranked world No. 4 in men's singles tennis by the Association of Tennis Professionals (ATP).[3] He has won 20 Grand Slam singles titles—the most in history for a male player—and has held the world No. 1 spot in the ATP rankings for a record total of 310 weeks (including a record 237 consecutive weeks) and was the year-end No. 1 five times, including four consecutive. Federer, who turned professional in 1998, was continuously ranked in the top 10 from October 2002 to November 2016."
    #     text_2 = "Federer has won a record eight Wimbledon men's singles titles, six Australian Open titles, five US Open titles (all consecutive, a record), and one French Open title. He is one of eight men to have achieved a Career Grand Slam. Federer has reached a record 31 men's singles Grand Slam finals, including 10 consecutively from the 2005 Wimbledon Championships to the 2007 US Open. Federer has also won a record six ATP Finals titles, 28 ATP Tour Masters 1000 titles, and a record 24 ATP Tour 500 titles. Federer was a member of Switzerland's winning Davis Cup team in 2014. He is also the only player after Jimmy Connors to have won 100 or more career singles titles, as well as to amass 1,200 wins in the Open Era."
    #
    #     processed = {"test_doc1": [text_1, []],
    #                  "test_doc2": [text_2, []],
    #                  "test_doc3": [text_1, []],
    #                  "test_doc4": [text_2, []],
    #                  "test_doc5": [text_1, []],
    #                  "test_doc6": [text_2, []],
    #                  "test_doc7": [text_1, []],
    #                  "test_doc8": [text_2, []],
    #                  "test_doc9": [text_1, []],
    #                  "test_doc10": [text_2, []],
    #                  }
    #     return processed
    #
    #
    # start = time.time()
    # for i in range(1):
    #
    #
    #     input_text = example_preprocessing()
    #
    #     mention_detection = MentionDetection(base_url, wiki_version)
    #     tagger_ner = load_flair_ner("ner-fast")
    #     # tagger_ngram = Cmns(base_url, wiki_version, n=5)
    #     mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)
    #
    #     config = {
    #         "mode": "eval",
    #         "model_path": model_path,
    #     }
    #
    #     model = EntityDisambiguation(base_url, wiki_version, config)
    #     predictions, timing = model.predict(mentions_dataset)
    #
    #     result = process_results(mentions_dataset, predictions, input_text)
    #     print('*** result ***')
    #     print(result)

    # print(time.time() - start)
