from Vocab import get_program, get_vocab

from FileSaver import save_file
from tracr_to_tlens import create_model_input, decode_model_output, convert_tracr_to_tl


if __name__ == '__main__':
    name_of_task = 'reverse'

    examples = get_vocab(name_of_task)
    tracr_model = get_program(name_of_task)
    tl_model = convert_tracr_to_tl(tracr_model)

    for data in examples:
        print("data: ", data)
        print("tracr model output",tracr_model.apply(data).decoded)

        input_tokens = create_model_input(
            data,
            tracr_model.input_encoder
        )

        logits, cache = tl_model.run_with_cache(input_tokens)
        decoded_output = decode_model_output(
            logits,
            tracr_model.output_encoder,
            tracr_model.input_encoder.bos_token
        )
        print("tl model output: ", decoded_output)

        save_file("patterns/parentheses.json", cache, tl_model)