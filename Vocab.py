from tracr.compiler import lib, compiling
from tracr.rasp import rasp


def get_vocab(program_name):
    match program_name:
        # TODO not working tl with length bigger then 5
        case "length":
            return [["BOS", 'a', 'b', 'd', 'c'],
                    ["BOS", 'd', 'b', 'b'],
                    ["BOS", 'a', 'b', 'a', 'b', 'a'],
                    ["BOS", 'c', 'b'], ]

        case "sort" | 'reverse':
            return [["BOS", 2, 5, 1, 3, 4],
                    ["BOS", 4, 3, 2, 1, 5],
                    ["BOS", 4, 2, 3, 5, 1],
                    ["BOS", 5, 4, 2, 1, 3]]

        # TODO tl not working
        case "dyck-3":
            return [
                ["BOS", "{", "}", "[", "]"],
                ["BOS", "(", ")", "[", "}"],
                ["BOS", "(", "{", "}", ')', ],
                ["BOS", "[", "(", '{', "[", "]"],
            ]
        # TODO sort_freq tl is returning wrong answers
        case 'sort_freq':
            return [
                ["BOS", "a", "c", "b", "c", "b"],
                ["BOS", "c", "b", "b"],
                ["BOS", "a", "a", "a", "b", "b"],
                ["BOS", "b", "c", "b", "c", "c"],
            ]

        case _:
            raise NotImplementedError(f"Vocab {program_name} not implemented.")


def make_reverse_program():
    def make_length():
        all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
        return rasp.SelectorWidth(all_true_selector)

    length = make_length()
    opp_index = length - rasp.indices - 1
    flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
    return rasp.Aggregate(flip, rasp.tokens)


def get_program(program_name, max_seq_len=6):
    if program_name == "length":
        vocab = {"a", "b", "c", "d"}
        program = lib.make_length()
    elif program_name == "dyck-3":
        vocab = {"(", ")", "{", "}", "[", "]"}
        program = lib.make_shuffle_dyck(pairs=["()", "{}", "[]"])
    elif program_name == "sort":
        vocab = {1, 2, 3, 4, 5}
        program = lib.make_sort(
            rasp.tokens, rasp.tokens, max_seq_len=max_seq_len, min_key=1)
    elif program_name == "reverse":
        vocab = {1, 2, 3, 4, 5}
        program = make_reverse_program()
    elif program_name == "sort_freq":
        vocab = {"a", "b", "c", "d"}
        program = lib.make_sort_freq(max_seq_len=max_seq_len)
    else:
        raise NotImplementedError(f"Program {program_name} not implemented.")
    return compiling.compile_rasp_to_model(
        program=program,
        vocab=vocab,
        max_seq_len=max_seq_len,
        causal=False,
        compiler_bos="BOS",
        compiler_pad="pad",
        mlp_exactness=100)
