import torch
import torch.nn as nn
import torch.nn.functional as F

from fastNLP.modules.utils import seq_mask
from fastNLP.models import BaseModel
from fastNLP.modules import encoder, decoder


def TZ(*args):
    return torch.zeros(*args).cuda()


class HyperLinear(nn.Module):
    def __init__(self, ninp, nout, nz=10):
        super(HyperLinear, self).__init__()
        # self.zW = nn.Linear(nz, ninp*nout)
        self.zb = nn.Linear(nz, nout)
        self.iz = nn.Linear(ninp, nz)
        self.zz = nn.Linear(nz, nz * ninp)
        self.zW = nn.Linear(nz, nout)
        self.ninp, self.nout, self.nz = ninp, nout, nz

    def forward(self, data):
        z = self.iz(data)
        W = self.zW(self.zz(z).view(-1, self.ninp, self.nz))
        b = self.zb(z)
        data_size = list(data.size()[:-1]) + [self.nout]
        return (torch.matmul(data.view(-1, 1, self.ninp), W) + b.view(-1, 1, self.nout)).view(*data_size)

    def fake_forward(self, data):
        z = self.iz(data)
        W = self.zW(z).view(-1, self.ninp, self.nout)
        b = self.zb(z)
        data_size = list(data.size()[:-1]) + [self.nout]
        return (torch.matmul(data.view(-1, 1, self.ninp), W) + b.view(-1, 1, self.nout)).view(*data_size)


class SLSTM(nn.Module):
    def __init__(self, nemb, nhid, num_layer, Tar_emb, hyper=False, dropout=0.5, return_all=True):
        super(SLSTM, self).__init__()
        if hyper:
            self.n_fc = nn.Linear(4 * nhid + nemb, 2 * nhid)
            self.n_h_fc = HyperLinear(4 * nhid + nemb, 5 * nhid)
            self.g_out_fc = nn.Linear(2 * nhid, nhid)
            self.g_att_fc = nn.Linear(2 * nhid, nhid)
        else:
            self.n_fc = nn.Linear(5 * nhid, 7 * nhid)
            self.g_out_fc = nn.Linear(2 * nhid, nhid)
            self.g_att_fc = nn.Linear(2 * nhid, nhid)

        self.emb = nn.Embedding.from_pretrained(Tar_emb)
        # self.emb.weight = nn.Parameter(Tar_emb)
        self.input = nn.Linear(nemb, nhid)
        self.fc = nn.Linear(2 * nhid, 2)
        self.up_fc = nn.Linear(nhid, 2 * nhid)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.MP_ITER = 9
        self.nemb, self.nhid, self.num_layer, self.hyper = nemb, nhid, num_layer, hyper

        self._return_all = return_all

    def forward(self, data, mask):
        # print("====================Start of forward======================")
        B, L = data.size()
        H = self.nhid

        def update_nodes(embs, nhs, ncs, gh, gc):
            ihs = torch.cat(
                [torch.cat([TZ(B, 1, H), nhs[:, :-1, :]], 1), nhs, torch.cat([nhs[:, 1:, :], TZ(B, 1, H)], 1), embs,
                 gh[:, None, :].expand(B, L, H)], 2)
            if self.hyper:
                xx = self.n_fc(ihs)
                og = F.sigmoid(xx[:, :, :self.nhid])
                uh = torch.tanh(xx[:, :, self.nhid:2 * self.nhid])
                fs = self.n_h_fc(ihs)
                gs = F.softmax(fs.view(embs.size(0), embs.size(1), 5, self.nhid), 2)
            else:
                fs = self.n_fc(ihs)
                og = F.sigmoid(fs[:, :, :self.nhid])
                uh = torch.tanh(fs[:, :, self.nhid:2 * self.nhid])
                gs = F.softmax(fs[:, :, self.nhid * 2:].view(embs.size(0), embs.size(1), 5, self.nhid), 2)

            # ics = torch.stack([ torch.cat([TZ(B,1,H), ncs[:,:-1,:]], 1), ncs, torch.cat([ncs[:,1:,:], TZ(B,1,H)], 1), gc[:,None,:].expand(B,L,H), uh], 2)
            ics = torch.stack(
                [torch.cat([TZ(B, 1, H), ncs[:, :-1, :]], 1), ncs, torch.cat([ncs[:, 1:, :], TZ(B, 1, H)], 1),
                 gc[:, None, :].expand(B, L, H), embs], 2)
            n_c = torch.sum(gs * ics, 2)
            n_nhs = og * torch.tanh(n_c)
            return n_nhs, n_c

        def update_g_node(nhs, ncs, gh, gc, mask):
            h_bar = nhs.sum(1) / mask.sum(1)[:, None]
            # h_bar = nhs.mean(1)
            ihs = torch.cat([h_bar[:, None, :], nhs], 1)
            ics = torch.cat([gc[:, None, :], ncs], 1)
            fs = self.g_att_fc(torch.cat([gh[:, None, :].expand(B, L + 1, H), ihs], 2))
            fs = fs + (1. - torch.cat([mask[:, :, None].expand(B, L, H), TZ(B, 1, H) + 1], 1)) * 200.0
            n_gc = torch.sum(F.softmax(fs, 1) * ics, 1)
            n_gh = F.sigmoid(self.g_out_fc(torch.cat([gh, h_bar], 1))) * torch.tanh(n_gc)
            return n_gh, n_gc

        embs = self.drop1(self.emb(data))
        embs = self.input(embs)
        # nhs = ncs = TZ(B,L,H)
        nhs = ncs = embs
        # gh = gc = TZ(B,H)
        gh = gc = embs.sum(1) / mask.sum(1)[:, None]
        for i in range(self.MP_ITER):
            n_gh, n_gc = update_g_node(nhs, ncs, gh, gc, mask)
            nhs, ncs = update_nodes(embs, nhs, ncs, gh, gc)
            nhs = mask[:, :, None].expand(B, L, H) * nhs
            ncs = mask[:, :, None].expand(B, L, H) * ncs
            gh, gc = n_gh, n_gc
        nhs = self.drop2(nhs)

        if self._return_all == False:
            return nhs

        rep = torch.cat([nhs, gh[:, None, :]], 1).sum(1) / mask.sum(1)[:, None]
        rep = self.drop2(rep)
        rep = torch.tanh(self.up_fc(rep))
        # rep = self.drop2(rep)
        pred = F.log_softmax(self.fc(rep), 1)

        # pred = F.log_softmax(self.drop2(self.fc(gh)), 1)
        # print("====================End of forward======================")

        return {'prediction': pred}


class SeqLabelingForSLSTM(BaseModel):
    """
    PyTorch Network for sequence labeling
    """

    def __init__(self, args):
        super(SeqLabelingForSLSTM, self).__init__()
        vocab_size = args["vocab_size"]
        word_emb_dim = args["word_emb_dim"]
        hidden_dim = args["rnn_hidden_units"]
        num_classes = args["num_classes"]
        init_emb = args["init_embedding"]
        print("num_classes:%d" % num_classes)

        #self.Embedding = encoder.Embedding(vocab_size, word_emb_dim)
        #self.Rnn = encoder.LSTM(word_emb_dim, hidden_dim)
        self.Rnn = SLSTM(word_emb_dim, hidden_dim, num_layer=1, Tar_emb=init_emb, return_all=False)
        self.Linear = encoder.Linear(hidden_dim, num_classes)
        self.Crf = decoder.CRF.ConditionalRandomField(num_classes)
        self.mask = None

    def forward(self, word_seq, word_seq_origin_len, truth=None):
        """
        :param word_seq: LongTensor, [batch_size, mex_len]
        :param word_seq_origin_len: LongTensor, [batch_size,], the origin lengths of the sequences.
        :param truth: LongTensor, [batch_size, max_len]
        :return y: If truth is None, return list of [decode path(list)]. Used in testing and predicting.
                    If truth is not None, return loss, a scalar. Used in training.
        """
        assert word_seq.shape[0] == word_seq_origin_len.shape[0]

        # print(word_seq.size())

        if truth is not None:
            assert truth.shape == word_seq.shape
        #self.mask = self.make_mask(word_seq, word_seq_origin_len)
        self.mask = word_seq_origin_len

        #x = self.Embedding(word_seq)
        # [batch_size, max_len, word_emb_dim]
        x = self.Rnn(word_seq, self.mask)
        # [batch_size, max_len, hidden_size * direction]
        x = self.Linear(x)
        # [batch_size, max_len, num_classes]
        #return {"loss": self._internal_loss(x, truth) if truth is not None else None,
        #        "predict": self.decode(x)}

        # loss = F.cross_entropy(x.view(-1, x.size(-1)), truth.view(-1)) if truth is not None else None
        # loss = loss.squeeze(0)
        # print(loss)
        # print(loss.size())

        return {"predict": x, "loss": F.cross_entropy(x.view(-1, x.size(-1)), truth.view(-1)) if truth is not None else None, "word_seq_origin_len":word_seq_origin_len.sum(1).long()}

    def predict(self, **kwargs):
        return self.forward(**kwargs)

    #def loss(self, x, y):
    #    """ Since the loss has been computed in forward(), this function simply returns x."""
    #    return x

    def _internal_loss(self, x, y):
        """
        Negative log likelihood loss.
        :param x: Tensor, [batch_size, max_len, tag_size]
        :param y: Tensor, [batch_size, max_len]
        :return loss: a scalar Tensor

        """
        x = x.float()
        y = y.long()
        assert x.shape[:2] == y.shape
        assert y.shape == self.mask.shape
        total_loss = self.Crf(x, y, self.mask)
        return torch.mean(total_loss)

    def make_mask(self, x, seq_len):
        batch_size, max_len = x.size(0), x.size(1)
        mask = seq_mask(seq_len, max_len)
        mask = mask.view(batch_size, max_len)
        mask = mask.to(x).float()
        return mask

    def decode(self, x, pad=True):
        """
        :param x: FloatTensor, [batch_size, max_len, tag_size]
        :param pad: pad the output sequence to equal lengths
        :return prediction: list of [decode path(list)]
        """
        max_len = x.shape[1]
        tag_seq = self.Crf.viterbi_decode(x, self.mask)
        # pad prediction to equal length
        if pad is True:
            for pred in tag_seq:
                if len(pred) < max_len:
                    pred += [0] * (max_len - len(pred))
        return tag_seq

