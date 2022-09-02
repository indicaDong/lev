import wandb
import torch
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from sacrebleu import corpus_bleu
from levenhtein_transformer.config import config
from levenhtein_transformer.utils import initialize_output_tokens, pad_tensors_in_dim
from levenhtein_transformer.layers import LevenshteinEncodeDecoder
from utils import vector_to_sentence
from get_shap_values import *


def validate(model: LevenshteinEncodeDecoder, iterator, SRC, TGT, EOS_WORD, eos, bos, pad, max_decode_iter=10,
             logging=False, is_test=False):
    model.eval()
    print('Validating...')
    # TODO: parallel Bleu calculation

    hypotheses_tokenized = []
    references_tokenized = []

    hypotheses = []
    references = []

    table = wandb.Table(columns=["Source", "Target", "Prediction"])

    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.trg

        src_sentences = [vector_to_sentence(src[i, :], SRC, EOS_WORD, start_from=0) for i in range(src.size(0))]
        tgt_sentences = [vector_to_sentence(tgt[i, :], TGT, EOS_WORD) for i in range(tgt.size(0))]
        print("src",src_sentences)
        print("tgt",tgt_sentences)
        decode_iter = 0
        prev_out = initialize_output_tokens(batch.trg, bos=bos, eos=eos)
        encoder_out = model.encode(batch.src, batch.src_mask)
        out = torch.tensor([[]])
        shap_values = None
        mask = torch.ones(batch.src.size(0), dtype=torch.bool)
        sentiment_scores = []
        
        while decode_iter < max_decode_iter:
            _out = model.decode(encoder_out[mask], prev_out[mask], batch.src_mask[mask],
                                max_ins_ratio=config['decoder_length_ratio'] if decode_iter == 0
                                else config['decoder_insertion_ratio'],
                                max_out_ratio=config['decoder_length_ratio'] , shap_values = shap_values)
            out, _out = pad_tensors_in_dim(prev_out, _out, dim=-1, pad=pad)
            out[mask] = _out

            prev_out, out = pad_tensors_in_dim(prev_out, out, dim=-1, pad=pad)
            # do not iterate over outputs that weren't changed
            mask = out.eq(prev_out).sum(-1) != out.size(-1)

            if out.tolist() == prev_out.tolist():
                break

            decode_iter += 1
            prev_out = out

            ##这里加入shap values 然后返回值给decode！
            print(out.size())
            shap_values, score = get_shap_values(out ,SRC)
            #sentiment_scores.append(score)

            
            
        ##plot sentiment scores
        #plot_sentiment(sentiment_scores)

        out_sentences = [vector_to_sentence(out[i, :], TGT, EOS_WORD) for i in range(out.size(0))]
        print("out_sentences",out_sentences)

        hypotheses_tokenized += [out_sentence.split(' ') for out_sentence in out_sentences]
        references_tokenized += [[tgt_sentence.split(' ')] for tgt_sentence in tgt_sentences]

        hypotheses += out_sentences
        references += tgt_sentences

        table.add_data(src_sentences[0], tgt_sentences[0], out_sentences[0])
        if logging:
            _src_sentences = [vector_to_sentence(src[i, :], SRC, EOS_WORD, start_from=0) for i in range(src.size(0))]
            _tgt_sentences = [vector_to_sentence(tgt[i, :], TGT, EOS_WORD) for i in range(tgt.size(0))]
            _out_sentences = [vector_to_sentence(out[i, :], TGT, EOS_WORD) for i in range(out.size(0))]
            print(f"Source: {_src_sentences[0]}\nTarget: {_tgt_sentences[0]}\nPrediction: {_out_sentences[0]}\n")

        # batch_bleu = [sentence_bleu([sentence_pair[0].split(' ')], sentence_pair[1].split(' '))
        #                     for sentence_pair in sentence_pairs]

        # batch_sacrebleu = [corpus_bleu(sentence_pair[1], sentence_pair[0]).score
        #                     for sentence_pair in sentence_pairs]

        # batch_bleus += batch_bleu
        # batch_sacrebleus += batch_sacrebleu

        # print(f'Batch {i} | Bleu: {np.array(batch_bleu).mean() * 100} | Sacrebleu: {np.array(batch_sacrebleu).mean()}')

    # bleu_score = np.array(batch_bleus).mean() * 100
    # sacrebleu_score = np.array(batch_sacrebleus).mean()
    corpus_sacrebleu_score = corpus_bleu(hypotheses, [references])
    corpus_bleu_score = nltk_corpus_bleu(references_tokenized, hypotheses_tokenized)
    wandb.log({"Test samples" if is_test else "Validation samples": table})
    print(f'Corpus bleu: {corpus_bleu_score * 100} | Corpus sacrebleu: {corpus_sacrebleu_score.score}')
    return corpus_sacrebleu_score.score
