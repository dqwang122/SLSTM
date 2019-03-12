import _pickle
import os

import torch
from fastNLP import Adam
from fastNLP import Trainer, Tester, CrossEntropyLoss, AccuracyMetric

from args import get_args
from model.SModel import SLSTM
from preprocess import load_dataset_with_glove

arg = get_args()
for k in arg.__dict__.keys():
    print(k, arg.__dict__[k])

save_dir = os.path.join("./save", "cls")
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu

if os.path.exists(f'./cache/{arg.dataset}_train_dataset.pkl'):
    train_dataset = _pickle.load(open(f'./cache/{arg.dataset}_train_dataset.pkl', 'rb'))
    dev_dataset = _pickle.load(open(f'./cache/{arg.dataset}_dev_dataset.pkl', 'rb'))
    embedding = _pickle.load(open(f'./cache/{arg.dataset}_embedding.pkl', 'rb'))
    test_dataset = _pickle.load(open(f'./cache/{arg.dataset}_test_dataset.pkl', 'rb'))
    vocab = _pickle.load(open(f'./cache/{arg.dataset}_vocab.pkl', 'rb'))
else:
    train_dataset, embedding, vocab = load_dataset_with_glove(data_dir=arg.data_dir, data_path=arg.dataset + '.task.train', glove_path=arg.glove_path)
    dev_dataset = load_dataset_with_glove(data_dir=arg.data_dir,data_path=arg.dataset + '.task.dev', load_glove=False, vocabs=vocab)
    test_dataset = load_dataset_with_glove(data_dir=arg.data_dir,data_path=arg.dataset + '.task.test', load_glove=False, vocabs=vocab)

    # dataset = combine_data_set(train_dataset, dev_dataset)

_pickle.dump(train_dataset, open(f'./cache/{arg.dataset}_train_dataset.pkl', 'wb'))
_pickle.dump(dev_dataset, open(f'./cache/{arg.dataset}_dev_dataset.pkl', 'wb'))
_pickle.dump(test_dataset, open(f'./cache/{arg.dataset}_test_dataset.pkl', 'wb'))
_pickle.dump(embedding, open(f'./cache/{arg.dataset}_embedding.pkl', 'wb'))
_pickle.dump(vocab, open(f'./cache/{arg.dataset}_vocab.pkl', 'wb'))


model = SLSTM(nemb=300, nhid=300, num_layer=1, Tar_emb=embedding)

trainer = Trainer(
    train_data=train_dataset,
    model=model,
    loss=CrossEntropyLoss(pred='predict', target='label'),
    metrics=AccuracyMetric(),
    n_epochs=20,
    batch_size=arg.batch_size,
    print_every=1,
    validate_every=-1,
    dev_data=dev_dataset,
    use_cuda=True,
    save_path=save_dir,
    optimizer=Adam(1e-3, weight_decay=0),
    check_code_level=-1,
    metric_key='acc',
    # sampler=default,
    use_tqdm=True,
)

results = trainer.train(load_best_model=True)
print(results)

torch.save(model, os.path.join(save_dir,"best_model.pkl"))


tester = Tester(
    data=test_dataset,
    model=model,
    metrics=AccuracyMetric(),
    batch_size=arg.batch_size,
    use_cuda=True,
)

eval_results = tester.test()
print(eval_results)
