from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jieba
import csv
import collections
import visdom
import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
import numpy as np

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

dataset = 'pheme'
# fold:0~4
data_dir = {'pheme':'./pheme_fold0.csv','weibo':'./weibo_fold0.csv'}

all_lines = []
source_ids = []
source_label = {}
with open(data_dir[dataset], newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_lines.append(row)
        if dataset == 'pheme':
            source_ids.append(row['source_id'])
            source_label[row['source_id']] = row['label']
        elif dataset == 'weibo':
            source_ids.append(row['event_id'])
            source_label[row['event_id']] = row['label']

PAD_IDX, UNK_IDX  = 0, 1
special_symbols = ['<unk>', '<pad>']

'''
if dataset == 'pheme':
    tokenizer = get_tokenizer('spacy', language='en_core_web_trf')
elif dataset == 'weibo':
    tokenizer = lambda x: jieba.lcut(x)
'''
tokenizer = lambda x: x.strip().split(' ')

def yield_tokens(data_iter):
    for contents in data_iter:
        yield tokenizer(contents['text'])

vocab = build_vocab_from_iterator(yield_tokens(all_lines), min_freq=1, specials=special_symbols, special_first=True)
vocab.set_default_index(UNK_IDX)

text_pipeline = lambda x: vocab(tokenizer(x))
if dataset == 'pheme':
    label_pipeline = lambda x: {'non-rumours':0,'rumours':1}[x]
elif dataset == 'weibo':
    label_pipeline = lambda x: int(x)

class myDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.id_counter = collections.Counter(source_ids)

        self.data, self.label, self.offset = self.preprocess()

    def preprocess(self):
        data_list, label_list, offset_list = [], [], []
        with open(data_dir[dataset], newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_list.append(text_pipeline(row['text']))
        start = 0
        for key, value in self.id_counter.items():
            label_list.append(label_pipeline(source_label[key]))
            offset_list.append((start, start+value))
            start += value
        return data_list, label_list, offset_list


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data, label, offset = self.data[self.offset[idx][0]:self.offset[idx][1]],\
                              self.label[idx], self.offset[idx]
        return data, label, offset

def collate_fn(batch):
    # for batch
    # N：number of source_id
    # dt: list
    # lb: scalar
    # offset: tuple
    data_for_word_pad,label,offset = [], [], []
    sen_len = []
    start = 0
    for dt, lb, offst in batch:
        sen_len.append(torch.LongTensor([len(d) for d in dt]))
        data_for_word_pad.extend([torch.LongTensor(d) for d in dt])
        flag = 0
        if offst[1] - offst[0] == 1:
            data_for_word_pad.extend([torch.LongTensor(d) for d in dt])
            flag = 1
        label.append(lb)
        offset.append((start, start+offst[1]-offst[0]+flag))
        start += offst[1]-offst[0]+flag
    data_tensor = pad_sequence(data_for_word_pad,batch_first=True,padding_value=PAD_IDX)
    data_for_sen_pad = []
    for offst in offset:
        data_for_sen_pad.append(data_tensor[offst[0]:offst[1]])
    data = pad_sequence(data_for_sen_pad,batch_first=True,padding_value=PAD_IDX)
    sen_len_pad = pad_sequence(sen_len,batch_first=True,padding_value=1)
    # sen_len_pad: [N, max_doc_len]
    # data: [N, max_doc_len, max_sen_len]
    label = torch.tensor(label, dtype=torch.long)
    offset = torch.tensor(offset, dtype=torch.long)
    # print(data.shape,label.shape,offset.shape)
    return data, label, offset, sen_len_pad

def get_dataloader(data_dir, batch_size, n_workers):
    dataset = myDataset(data_dir)
    train_idx = range(int(0.72 * len(dataset)))
    valid_idx = range(int(0.72 * len(dataset)), int(0.9 * len(dataset)))
    test_idx = range(int(0.9 * len(dataset)), len(dataset))
    trainset, validset, testset = Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx)
    print(len(trainset), len(validset), len(testset))
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=n_workers,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True,
                              collate_fn=collate_fn
                              )
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              num_workers=n_workers,
                              drop_last=False,
                              pin_memory=True,
                              collate_fn=collate_fn
                              )
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             num_workers=n_workers,
                             drop_last=False,
                             pin_memory=True,
                             collate_fn=collate_fn
                             )
    return train_loader, valid_loader, test_loader

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)

    def forward(self, tokens):
        '''
        :param tokens: [b, max_doc_len, max_sen_len]
        :return:  [b, max_doc_len, max_sen_len, embedding_dim]
        '''
        return self.embedding(tokens)


class Model_Generator(nn.Module):
    def __init__(self, token_embedding, embedding_dim, hidden_dim, max_sen_len_for_all, dropout=0.5):
        super(Model_Generator, self).__init__()
        self.emb = token_embedding
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2,
                          bidirectional=True, dropout=0.5)
        self.decov = nn.ConvTranspose1d(1, max_sen_len_for_all, 3, 1)
        self.fc = nn.Linear(hidden_dim*2+2, embedding_dim)

    def forward(self, input, offset, sen_len_pad):
        '''
        :param input: [b, max_doc_len, max_sen_len_for_batch]
        :param offset: [b, 2]
        :param sen_len_pad: [b, max_doc_len]
        :return:
        '''
        # b
        doc_len = offset[:,1]-offset[:,0]
        # [b, max_doc_len]
        sorted_lengths, idx = torch.sort(doc_len, dim=0, descending=True)
        _, recover_idx = torch.sort(idx, dim=0)
        # [b, max_doc_len, max_sen_len_for_batch] => [b, max_doc_len, max_sen_len_for_batch, embedding_dim]
        word_emb = self.emb(input)
        # word_emb = self.dropout(word_emb)
        # [b, max_doc_len, max_sen_len_for_batch, embedding_dim] => [b, max_doc_len, embedding_dim]
        doc_emb = word_emb.sum(2) / sen_len_pad.unsqueeze(2)

        # [b, max_doc_len, embedding_dim] => [max_doc_len-1, b, embedding_dim]
        sorted_embedding = doc_emb.index_select(dim=0, index=idx).transpose(0, 1)[1:]
        modify_doc_len = sorted_lengths - torch.where(sorted_lengths==1,0,1)
        sorted_embedding = nn.utils.rnn.pack_padded_sequence(sorted_embedding, batch_first=False,
                                                             lengths=modify_doc_len.cpu())

        # output: [max_doc_len-1, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        output, hidden = self.gru(sorted_embedding)
        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2] => [[b, 1, hid_dim*2]]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = hidden.index_select(dim=0, index=recover_idx).unsqueeze(1)
        # [b, 1, hid_dim*2] => [b, max_sen_len_for_all, embedding_dim]
        out = self.fc(self.decov(hidden))

        max_sen_len_for_batch = input.shape[-1]
        # [b, max_sen_len_for_all, embedding_dim] => [b, max_sen_len_for_batch, embedding_dim]
        out = out[:,:max_sen_len_for_batch]

        mask = input[:,0] != 0 # mask: [b, max_sen_len_for_batch]
        # out: [b, max_sen_len_for_batch, embedding_dim]
        out = out * mask.unsqueeze(2)

        return out

class Model_Discriminator(nn.Module):
    def __init__(self, dropout, embedding_dim):
        super(Model_Discriminator, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.cov1 = nn.Conv2d(1, 128, kernel_size=[2, embedding_dim], stride=1)
        self.cov2 = nn.Conv2d(1, 128, kernel_size=[3, embedding_dim], stride=1)
        self.cov3 = nn.Conv2d(1, 128, kernel_size=[4, embedding_dim], stride=1)

        self.fc = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(32, 2)
        self.fc2 = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        '''
        :param input: [b, max_sen_len_for_batch, embedding_dim] 所有的source
        :return:
        '''
        input = self.dropout(input)
        # [b, max_sen_len_for_batch, embedding_dim] => [b, 1, max_sen_len_for_batch, embedding_dim]
        x = input.unsqueeze(1)
        # [b,128] 128个out_channel
        out1 = F.adaptive_max_pool2d(self.cov1(x), [1, 1]).squeeze()
        # [b,128]
        out2 = F.adaptive_max_pool2d(self.cov2(x), [1, 1]).squeeze()
        # [b,128]
        out3 = F.adaptive_max_pool2d(self.cov3(x), [1, 1]).squeeze()
        # [b, 384]
        out = torch.cat([out1,out2,out3], dim=1)

        # [b, 32]
        out = self.fc(out)

        # [b, 32] => [b, 2]
        class_logits = self.fc1(out)
        # [b, 32] => [b, 1]
        probability = self.fc2(out)

        return class_logits, probability

def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.)

def gradient_penalty(device, batchsz, D, xr, xf):
    """

    :param D:
    :param xr: [b,max_sen_len_for_batch, embedding_dim]
    :param xf: [b,max_sen_len_for_batch, embedding_dim]
    :return:
    """

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1, 1] => [b, max_sen_len_for_batch, embedding_dim]
    alpha = torch.rand(batchsz, 1, 1).to(device)
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)[1]

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=(1,2)) - 1) ** 2).mean()

    return gp

def evalute(token_embedding, model_D, device, criteon, data_loader, viz):
    model_D.eval()

    correct = 0
    total_loss = 0
    total = len(data_loader.dataset)
    target_names = ['NR', 'R']
    all_y = []
    all_pred = []
    with torch.no_grad():
        for x, y, offset, sen_len_pad in data_loader:
            x, y, offset, sen_len_pad = x.to(device), y.to(device), offset.to(device), sen_len_pad.to(device)

            xr = token_embedding(x)[:, 0]
            class_logits_r, probability_r = model_D(xr)
            class_loss_r = criteon(class_logits_r, y)

            total_loss += class_loss_r.item()
            pred = class_logits_r.argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()

            all_y.extend(y.detach().cpu().tolist())
            all_pred.extend(pred.detach().cpu().tolist())

        viz.text(str(np.array(all_y)), win='label', opts=dict(title='label'))
        viz.text(str(np.array(all_pred)), win='pred', opts=dict(title='pred'))

        report = classification_report(all_y, all_pred, target_names=target_names, digits=5)

    return correct / total, total_loss / total, report


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50
    batch_size = 64 #pheme 64 weibo 16
    d_steps = 5
    lr = 1e-3
    lambda_gp = 0.3
    n_workers = 8
    embedding_dim = 300
    hidden_dim = 256
    max_sen_len_for_all = 115 if dataset == 'pheme' else 140
    vocab_size = len(vocab)
    print(vocab_size)

    viz = visdom.Visdom()
    viz.line([[0.0, 0.0]], [0.], win='acc', opts=dict(title='train&val acc', legend=['train', 'val']))
    viz.line([[0.0, 0.0]], [0.], win='loss', opts=dict(title='train&val loss', legend=['train', 'val']))
    global_step = 0

    train_loader, valid_loader, test_loader = get_dataloader(data_dir, batch_size, n_workers)

    token_embedding = TokenEmbedding(vocab_size,embedding_dim).to(device)
    G = Model_Generator(token_embedding,embedding_dim,hidden_dim,max_sen_len_for_all).to(device)
    D = Model_Discriminator(dropout=0.4, embedding_dim=embedding_dim).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    criteon = nn.CrossEntropyLoss().to(device)

    best_acc = 0
    for epoch in range(epochs):
        # 1. train discriminator for k steps
        for d in range(d_steps):
            train_loss_D = 0
            train_correct = 0
            train_f_correct = 0
            for step, (x, y, offset, sen_len_pad) in enumerate(train_loader):
                x, y, offset, sen_len_pad = x.to(device), y.to(device), offset.to(device), sen_len_pad.to(device)
                # print(x.shape,y.shape,offset.shape,sen_len_pad.shape)

                D.train()
                # real data
                # [b, max_doc_len, max_sen_len_for_batch] => [b, max_doc_len, max_sen_len_for_batch, embedding_dim]
                # => [b,max_sen_len_for_batch, embedding_dim]
                xr = token_embedding(x)[:,0]
                # [b, 2], [b, 1]
                class_logits_r, probability_r = D(xr)
                class_loss_r = criteon(class_logits_r, y)
                loss_r = -(probability_r.mean())

                # produce fake data
                # [b,max_sen_len_for_batch, embedding_dim]
                G.eval()
                xf = G(x, offset, sen_len_pad).detach()
                class_logits_f, probability_f = D(xf)
                class_loss_f = criteon(class_logits_f, y)
                loss_f = probability_f.mean()

                gp = lambda_gp * gradient_penalty(device, x.shape[0], D, xr, xf)

                loss_D = class_loss_r + class_loss_f + loss_r + loss_f + gp
                train_loss_D += loss_D.item()

                # [b]
                pred_label = class_logits_r.argmax(dim=1)
                train_correct += pred_label.eq(y).float().sum().item()

                pred_f_label = class_logits_f.argmax(dim=1)
                train_f_correct += pred_f_label.eq(y).float().sum().item()

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            print(f'epoch={epoch},d_step={d},train loss_D={(train_loss_D / len(train_loader.dataset)):.6f},'
                  f'train acc={(train_correct / len(train_loader.dataset)):.4f},'
                  f'train f acc={(train_f_correct / len(train_loader.dataset)):.4f}')

        train_loss_G = 0
        # 2. train Generator
        for step, (x, y, offset, sen_len_pad) in enumerate(train_loader):
            x, y, offset, sen_len_pad = x.to(device), y.to(device), offset.to(device), sen_len_pad.to(device)

            G.train()
            xf = G(x, offset, sen_len_pad)
            class_logits_f, probability_f = D(xf)
            loss_f = probability_f.mean()
            class_loss_f = criteon(class_logits_f, y)

            loss_G = class_loss_f - loss_f
            train_loss_G += loss_G.item()

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        print(f'epoch={epoch},train loss_G={(train_loss_G / len(train_loader.dataset)):.6f}')

        if epoch % 1 == 0:

            val_acc, val_loss, report = evalute(token_embedding, D, device, criteon, valid_loader, viz)
            print(f'val_loss={val_loss:.6f},val acc={val_acc:.4f}')
            # print(report)
            global_step += 1
            viz.line([[train_correct / len(train_loader.dataset),val_acc]], [global_step], win='acc', update='append')
            viz.line([[train_loss_D / len(train_loader.dataset),val_loss]], [global_step], win='loss',update='append')

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(G.state_dict(), 'model_G_fold0.mdl')
                with open('report.txt', 'w') as f:
                    f.write(report)


    print(f'best val acc:{best_acc}', f'best epoch:{best_epoch}')

if __name__ == '__main__':
    main()