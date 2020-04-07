
import os

def write_topics_from_qrels(qrels_path, topics_path):
    """ Given a TREC standard QRELS file in 'qrels_path', write TREC standard TOPICS file in 'file_path'. """
    # Store queries already written to file.
    written_queries = []
    with open(topics_path, 'w') as topics_f:
        with open(qrels_path, 'r') as qrels_f:
            for line in qrels_f:
                # Extract query from QRELS file.
                query, _, _, _ = line.split(' ')
                if query not in written_queries:
                    # Write query to TOPICS file.
                    topics_f.write(query + '\n')
                    # Add query to 'written_queries' list.
                    written_queries.append(query)


if __name__ == '__main__':
    qrels_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.qrels')
    topics_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..')), 'data', 'test.pages.cbor-hierarchical.entity.topics')

    write_topics_from_qrels(qrels_path=qrels_path, topics_path=topics_path)