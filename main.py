
import os

from learning.experiments import FineTuningReRanking


if __name__ == '__main__':

    train_data_dir_path = 'nfs/trec_car/data/bert_reranker_datasets/test_chunks_train/'
    train_batch_size = 8
    dev_data_dir_path = 'nfs/trec_car/data/bert_reranker_datasets/test_chunks/'
    dev_batch_size = 32

    experiment = FineTuningReRanking(train_data_dir_path=train_data_dir_path,
                                     train_batch_size=train_batch_size,
                                     dev_data_dir_path=dev_data_dir_path,
                                     dev_batch_size=dev_batch_size)

    epochs = 3
    lr = 2e-5
    eps = 1e-8
    weight_decay = 0.01
    num_warmup_steps = 0
    seed_val = 42
    experiments_dir = '/nfs/trec_car/data/bert_reranker_datasets/exp/'
    experiment_name = 'test_exp_2'
    write = True
    logging_steps = 30

    experiment.run_experiment(epochs=epochs,
                              lr=lr,
                              eps=eps,
                              weight_decay=weight_decay,
                              num_warmup_steps=num_warmup_steps,
                              seed_val=seed_val,
                              experiments_dir=experiments_dir,
                              experiment_name=experiment_name,
                              write=write,
                              logging_steps=logging_steps)






