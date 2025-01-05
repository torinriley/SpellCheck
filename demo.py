import torch
from model import build_transformer

character_vocab = {char: idx for idx, char in enumerate(
    ['<pad>', '<sos>', '<eos>', '<unk>'] + [chr(i) for i in range(97, 123)])}
idx_to_char = {idx: char for char, idx in character_vocab.items()}

pad_idx = character_vocab['<pad>']
sos_idx = character_vocab['<sos>']
eos_idx = character_vocab['<eos>']

src_seq_len = tgt_seq_len = 20
d_model = 256
num_layers = 6
num_heads = 8
dropout = 0.1
d_ff = 1024

def load_model():
    model = build_transformer(
        src_vocab_size=len(character_vocab),
        tgt_vocab_size=len(character_vocab),
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        d_ff=d_ff
    )
    model.load_state_dict(torch.load("spell_check_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def tokenize_word(word):
    return [sos_idx] + [character_vocab.get(c, character_vocab['<unk>']) for c in word] + [eos_idx]

def decode_sequence(indices):
    chars = [idx_to_char[idx] for idx in indices if idx not in [pad_idx, sos_idx, eos_idx]]
    return ''.join(chars)

def beam_search_decode(encoder_output, src_mask, model, beam_width=3, max_len=20):
    device = encoder_output.device
    start_token = torch.tensor([[sos_idx]], device=device)
    end_token = eos_idx

    sequences = [(start_token, 0)]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            tgt_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
            decoder_output = model.decode(encoder_output, src_mask, seq, tgt_mask)
            predictions = model.project(decoder_output[:, -1, :])
            top_k_probs, top_k_indices = predictions.topk(beam_width)

            for i in range(beam_width):
                candidate_seq = torch.cat([seq, top_k_indices[:, i].unsqueeze(0)], dim=1)
                candidate_score = score + top_k_probs[0, i].item()
                all_candidates.append((candidate_seq, candidate_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if all(seq[0][0, -1].item() == end_token for seq in sequences):
            break

    best_sequence = sequences[0][0].squeeze(0).tolist()
    return decode_sequence(best_sequence[1:-1])

def infer(word, model, beam_width=3):
    tokenized = tokenize_word(word)
    tokenized += [pad_idx] * (src_seq_len - len(tokenized))

    src = torch.tensor(tokenized).unsqueeze(0)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        encoder_output = model.encode(src, src_mask)
        corrected_word = beam_search_decode(encoder_output, src_mask, model, beam_width=beam_width)
    return corrected_word

if __name__ == "__main__":
    print("Loading model...")
    model = load_model()

    while True:
        word = input("Enter a misspelled word (or 'quit' to exit): ").strip().lower()
        if word == 'quit':
            break
        corrected_word = infer(word, model, beam_width=3)
        print(f"Corrected word: {corrected_word}")
