from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jieba
import csv
import collections
import visdom
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
import numpy as np
import time

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

class myTransformer(nn.Module):
    def __init__(self,embedding_dim, num_heads, num_layers, dropout=0.45):
        super(myTransformer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

    def forward(self, xr, xf, xr_padding_mask):
        '''
        :param xr: [b,max_sen_len_for_batch, embedding_dim]
        :param xf: [b,max_sen_len_for_batch, embedding_dim]
        :return:
        '''

        # [b,max_sen_len_for_batch, embedding_dim] => [b, embedding_dim]
        xr_out = self.dropout(xr)
        # [b,max_sen_len_for_batch, embedding_dim]
        xr_out = self.encoder(xr_out, src_key_padding_mask=xr_padding_mask)
        # [b,1,max_sen_len_for_batch, embedding_dim]
        xr_out = xr_out.unsqueeze(1)

        # [b,128] 128个out_channel
        xr_out1 = F.adaptive_max_pool2d(self.cov1(xr_out), [1, 1]).squeeze()
        # [b,128]
        xr_out2 = F.adaptive_max_pool2d(self.cov2(xr_out), [1, 1]).squeeze()
        # [b,128]
        xr_out3 = F.adaptive_max_pool2d(self.cov3(xr_out), [1, 1]).squeeze()
        # [b, 384]
        xr_out = torch.cat([xr_out1, xr_out2, xr_out3], dim=1)

        # [b, 32]
        xr_out = self.fc(xr_out)

        # # [b,max_sen_len_for_batch, embedding_dim] => [b, embedding_dim]
        xf_out = self.dropout(xf)
        # # [b,max_sen_len_for_batch, embedding_dim]
        # xf_out = self.encoder(xf, src_key_padding_mask=None)
        # # [b,1,max_sen_len_for_batch, embedding_dim]
        xf_out = xf_out.unsqueeze(1)

        # [b,128] 128个out_channel
        xf_out1 = F.adaptive_max_pool2d(self.cov1(xf_out), [1, 1]).squeeze()
        # [b,128]
        xf_out2 = F.adaptive_max_pool2d(self.cov2(xf_out), [1, 1]).squeeze()
        # [b,128]
        xf_out3 = F.adaptive_max_pool2d(self.cov3(xf_out), [1, 1]).squeeze()
        # [b, 384]
        xf_out = torch.cat([xf_out1, xf_out2, xf_out3], dim=1)

        # [b, 32]
        xf_out = self.fc(xf_out)

        # [b, 32] => [b, 2]
        logits_r = self.fc1(xr_out)
        logits_f = self.fc1(xf_out)

        return logits_r,logits_f


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.)


def evaluate_trans(token_embedding, model, G, device, criteon, data_loader,viz):
    model.eval()

    correct = 0
    total_loss = 0
    total = len(data_loader.dataset)
    target_names = ['NR', 'R']
    all_y = []
    all_pred = []
    with torch.no_grad():
        for x, y, offset, sen_len_pad in data_loader:
            x, y, offset, sen_len_pad = x.to(device), y.to(device), offset.to(device), sen_len_pad.to(device)

            xr_padding_mask = (x[:, 0] == PAD_IDX)
            xr = token_embedding(x)[:, 0]

            G.eval()
            xf = G(x, offset, sen_len_pad).detach()

            logits_r,logits_f = model(xr, xf, xr_padding_mask)
            loss_r = criteon(logits_r, y)
            loss_f = criteon(logits_f, y)
            loss = loss_r + 0.1 * loss_f

            total_loss += loss.item()
            pred = (logits_r + 0.1*logits_f).argmax(dim=1)
            correct += torch.eq(pred, y).sum().float().item()

            all_y.extend(y.detach().cpu().tolist())
            all_pred.extend(pred.detach().cpu().tolist())

        viz.text(str(np.array(all_y)), win='label', opts=dict(title='label'))
        viz.text(str(np.array(all_pred)), win='pred', opts=dict(title='pred'))

        report = classification_report(all_y, all_pred, target_names=target_names, digits=5)

        return correct / total, total_loss / total, report


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 50 #pheme 100
    batch_size = 64 # pheme 64
    lr = 1e-3
    n_workers = 8
    embedding_dim = 300
    hidden_dim = 256
    max_sen_len_for_all = 115 if dataset == 'pheme' else 140
    vocab_size = len(vocab)

    viz = visdom.Visdom()
    viz.line([[0.0, 0.0]], [0.], win='acc', opts=dict(title='train&val acc', legend=['train', 'val']))
    viz.line([[0.0, 0.0]], [0.], win='loss', opts=dict(title='train&val loss', legend=['train', 'val']))
    global_step = 0

    train_loader, valid_loader, test_loader = get_dataloader(data_dir, batch_size, n_workers)

    token_embedding = TokenEmbedding(vocab_size,embedding_dim).to(device)
    G = Model_Generator(token_embedding,embedding_dim,hidden_dim,max_sen_len_for_all).to(device)

    criteon = nn.CrossEntropyLoss()

    G.load_state_dict(torch.load('model_G_fold0.mdl'))
    model_trans = myTransformer(embedding_dim, num_heads=3, num_layers=1).to(device)
    optimizer = optim.Adam(model_trans.parameters(), lr=lr, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95,verbose=False)

    best_acc = 0
    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0
        start = time.time()
        for step, (x, y, offset, sen_len_pad) in enumerate(train_loader):
            x, y, offset, sen_len_pad = x.to(device), y.to(device), offset.to(device), sen_len_pad.to(device)

            G.eval()
            xf = G(x, offset, sen_len_pad).detach()

            xr_padding_mask = (x[:, 0] == PAD_IDX)
            xr = token_embedding(x)[:, 0]

            model_trans.train()
            logits_r,logits_f = model_trans(xr, xf, xr_padding_mask)
            loss_r = criteon(logits_r, y)
            loss_f = criteon(logits_f, y)
            loss = loss_r + 0.1 * loss_f

            train_loss += loss.item()
            pred_label = (logits_r + 0.1*logits_f).argmax(dim=1)
            train_correct += pred_label.eq(y).float().sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch={epoch},train loss={(train_loss / len(train_loader.dataset)):.6f},'
              f'train acc={(train_correct / len(train_loader.dataset)):.4f},')

        if epoch % 1 == 0:

            val_acc, val_loss, report = evaluate_trans(token_embedding, model_trans, G, device, criteon, valid_loader, viz)
            # print(f'val_loss={val_loss:.6f},val acc={val_acc:.4f}')

            # print(f'time={time.time()-start:.2f}')

            global_step += 1
            viz.line([[train_correct / len(train_loader.dataset), val_acc]], [global_step], win='acc', update='append')
            viz.line([[train_loss / len(train_loader.dataset), val_loss]], [global_step], win='loss', update='append')

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model_trans.state_dict(), 'best_pheme_fold0.mdl')

            print(f'val_loss={val_loss:.6f},val acc={val_acc:.4f},best_acc={best_acc:.4f}')
            print(report)

        scheduler.step()

    print(f'best val acc:{best_acc}', f'best epoch:{best_epoch}')
    model_trans.load_state_dict(torch.load('best_pheme_fold0.mdl'))
    print('loaded from ckpt!')
    test_report = evaluate_trans(token_embedding, model_trans, G, device, criteon, test_loader, viz)[2]
    print(test_report)

if __name__ == '__main__':
    main()