# %%
import itertools
import numpy as np
import torch
from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

from data_saver import save_file
from tracr_to_tlens import create_model_input, decode_model_output, convert_tracr_to_tl
from tracr_visual import plot_residuals_and_input, plot_layer_outputs


def make_reverse_program():
    """Creates a RASP program that reverses lists."""

    def make_length():
        all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        return rasp.SelectorWidth(all_true_selector)

    length = make_length()
    opp_index = length - rasp.indices - 1
    flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
    return rasp.Aggregate(flip, rasp.tokens)


def compile_reverse_model(max_seq_len=5, vocab={1, 2, 3}, bos="BOS"):
    """Compiles the reverse program into a Tracr model."""
    reverse = make_reverse_program()
    model = compiling.compile_rasp_to_model(
        reverse,
        vocab=vocab,
        max_seq_len=max_seq_len,
        compiler_bos=bos,
    )
    return model


def compile_sort_model(max_seq_len=5, vocab={1, 2, 3, 4, 5}, bos="BOS"):
    """Compiles the reverse program into a Tracr model."""
    program = lib.make_sort_unique(rasp.tokens, rasp.tokens)

    return compiling.compile_rasp_to_model(
        program=program,
        vocab=vocab,
        max_seq_len=max_seq_len,
        causal=False,
        compiler_bos="BOS",
        mlp_exactness=100)


def compile_length_model():
    vocab = {"a", "b", "c", "d"}
    program = lib.make_length()
    return compiling.compile_rasp_to_model(
        program=program,
        vocab=vocab,
        max_seq_len=max_seq_len,
        causal=False,
        compiler_bos="bos",
        compiler_pad="pad",
        mlp_exactness=100)


class ReverseDataset(torch.utils.data.Dataset):
    def __init__(self, max_seq_len, vocab_size):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        sequences = []
        labels = []
        # for length in range(2, self.max_seq_len+1):
        length = self.max_seq_len  # otherwise we would need padding
        # Generate all possible sequences of given length
        for seq in itertools.product(range(1, self.vocab_size + 1), repeat=length):
            input_seq = ["BOS"] + list(seq)
            target_seq = ["BOS"] + list(reversed(seq))
            sequences.append(input_seq)
            labels.append(target_seq)
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class EncodedReverseDataset(ReverseDataset):
    def __init__(self, max_seq_len, vocab_size, input_encoder, output_encoder):
        super().__init__(max_seq_len, vocab_size)
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.input_tokens = [create_model_input(sequence, input_encoder) for sequence in self.sequences]
        self.target_tokens = [create_model_input(label[1:], output_encoder) for label in self.labels]

    def __getitem__(self, idx):
        return self.input_tokens[idx].squeeze(0), self.target_tokens[idx].squeeze(0)


# %%

if __name__ == "__main__":
    max_seq_len = 4
    vocab_size = 4

    model = compile_reverse_model(
        max_seq_len=max_seq_len,
        vocab={i for i in range(0, vocab_size + 1)},
        bos="BOS"
    )

    dataset = ReverseDataset(max_seq_len=max_seq_len, vocab_size=vocab_size)
    print("dataset size: ", len(dataset))

    # random_idx = np.random.randint(0, len(dataset))
    random_idx = 143
    inputs, labels = dataset[random_idx]
    inputs = inputs[:-1]
    labels = labels[1:]
    print("inputs: ", inputs)
    print("labels: ", labels)

    out = model.apply(inputs)
    print(out.decoded)


    plot_residuals_and_input(
        model=model,
        inputs=inputs,
        figsize=(10, 9)
    )
    plot_layer_outputs(
        model=model,
        inputs=inputs,
        figsize=(8, 9)
    )
    import matplotlib.pyplot as plt

    plt.show()

    # %%
    # Usage:
    max_seq_len = 5
    vocab_size = 5

    tracr_model = compile_reverse_model(
        max_seq_len=max_seq_len,
        vocab={i for i in range(0, vocab_size + 1)},
        bos="BOS"
    )
    tl_model = convert_tracr_to_tl(tracr_model)

    dataset = EncodedReverseDataset(
        max_seq_len=5,
        vocab_size=5,
        input_encoder=tracr_model.input_encoder,
        output_encoder=tracr_model.output_encoder
    )

    # Now you can use it directly with the transformer:
    random_idx = np.random.randint(0, len(dataset))
    input_tokens, target_tokens = dataset[random_idx]
    print("input_tokens: ", input_tokens)
    print("target_tokens: ", target_tokens)
    logits, cache = tl_model.run_with_cache(input_tokens)
    print("logits.shape: ", logits.shape)
    decoded_output = decode_model_output(
        logits,
        tracr_model.output_encoder,
        tracr_model.input_encoder.bos_token
    )
    print("decoded_output: ", decoded_output)

    save_file("patterns/patterns2.json", cache, tl_model)
