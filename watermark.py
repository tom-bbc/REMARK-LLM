################################################################################################
# Imports
################################################################################################

import copy
import os
import random
import re
import itertools

import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch

from model import Mapping, TransformerClassifier
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Config,
)


################################################################################################
# Helpwer functions for watermarking process
################################################################################################

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def add_gumbel_noise(model, logits, input_idx, temperature):
    logits_prob = F.gumbel_softmax(logits, tau=temperature)
    cur_logit = F.gumbel_softmax(logits, tau=temperature)
    # decode
    return cur_logit, logits_prob


def beam_search_with_gumbel(
    model, input_ids, logits_seq, length, beam_width, temperature=1.0
):
    # Start with an empty sequence and score of 0.0 (since we'll be summing log probs)
    device = get_device()

    sequences = [
        (torch.tensor([]).long().to(device), 0.0)
    ]

    logits_seq = logits_seq[0]
    count = 0

    for logits in logits_seq:
        all_candidates = []
        log_noise_probs, log_probs = add_gumbel_noise(
            model, logits, input_ids[0][count], temperature
        )

        for seq, seq_log_probs in sequences:
            top_log_probs, top_ix = torch.topk(log_noise_probs, beam_width)

            for i in range(beam_width):
                next_seq = torch.cat([seq, top_ix[i].unsqueeze(0).long()], dim=-1)
                new_log_prob = seq_log_probs + top_log_probs[i]
                all_candidates.append((next_seq, new_log_prob))

        # Sort all candidates by the sum of their log probabilities and keep top beam_width sequences
        sorted_candidates = sorted(
            all_candidates, key=lambda x: x[1].sum(), reverse=True
        )

        sequences = sorted_candidates[:beam_width]

        if len(sequences[0][0]) == length:
            break

        count += 1

    sequences = [sequences[i][0] for i in range(len(sequences))]

    # Return the top sequence
    return sequences


def get_sample(model, input_ids, logits, beam_width=5, temperature=1, sentence=2):
    # In every decoding step, a Gumbel-Softmax noise with temperature τ is added into the token distribution Si.
    # Then, the beam search algorithm with beam size B produces B candidate sentences from the perturbed token distribution.
    length = input_ids.sum(dim=1)
    decoded_ids = []

    for _ in range(sentence):
        model_output = beam_search_with_gumbel(
            model, input_ids, logits, length=length, beam_width=beam_width, temperature=temperature
        )
        decoded_ids.extend(model_output)

    decoded_ids = torch.stack(decoded_ids, dim=0)

    return decoded_ids


def mask_input_ids(ids, tokenizer, mask_per):
    mask_token = tokenizer.unk_token_id
    new_idx = copy.deepcopy(ids)
    non_zero_idx = torch.nonzero(ids)
    random_idx = random.sample(
        range(non_zero_idx.shape[0]), int(mask_per * non_zero_idx.shape[0])
    )
    new_idx[non_zero_idx[random_idx, 0], non_zero_idx[random_idx, 1]] = mask_token

    return new_idx


################################################################################################
# Load watermarking and detection models and setup environment
################################################################################################

def setup_models(model_load_path, base_model_name, message_max_length:int = 16):
    # Set input parameters
    d_model = 512
    nhead = 8
    num_layers = 3
    num_classes = message_max_length

    device = get_device()

    # Load the model from the model_load_path
    config = T5Config.from_pretrained(base_model_name)
    config.message_max_length = message_max_length
    config.wm_embed_model = "t5"
    config.model_max_length = d_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, model_max_length=d_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_ckpt_path = os.path.join(model_load_path, "model.pt")
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, config=config)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device))

    extractor_ckpt_path = os.path.join(model_load_path, "extractor.pt")
    extractor = TransformerClassifier(d_model, nhead, num_layers, num_classes)
    extractor.load_state_dict(torch.load(extractor_ckpt_path, map_location=device))

    mapper_ckpt_path = os.path.join(model_load_path, "mapper.pt")
    mapper = Mapping(config.vocab_size, d_model)
    mapper.load_state_dict(torch.load(mapper_ckpt_path, map_location=device))

    model, extractor, mapper = model.to(device), extractor.to(device), mapper.to(device)

    return model, tokenizer, extractor, mapper


################################################################################################
# Watermark a text with a binary message
################################################################################################

def watermark(
    input_text: str, input_message: list,
) -> str:
    # Set input parameters
    model_load_path = "./logs_text_wm/model"
    base_model_name = "t5-base"
    message_max_length = 16
    mask_per = 0.3

    device = get_device()

    # Set random seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load watermarking models
    model, tokenizer, extractor, mapper = setup_models(
        model_load_path,
        base_model_name,
        message_max_length
    )

    # Load a single text and message sample from the dataset
    encoded_sequence = tokenizer(input_text, return_tensors="pt")
    encoded_sequence = encoded_sequence.to(device)
    input_ids = encoded_sequence.input_ids
    attention_mask = encoded_sequence.attention_mask

    input_sequence_quotes = re.findall('"([^"]*)"', input_text)

    # Format binary input message
    message_base = torch.tensor([input_message]).to(device)
    input_match_bit_list = []

    input_dist = F.one_hot(
        input_ids, num_classes=model.config.vocab_size
    ).float()

    with torch.no_grad():
        input_ebd = mapper(input_dist)
        input_logits = extractor(input_ebd)

    input_logits = input_logits > 0.5
    input_match_bit = torch.sum(input_logits == message_base, dim=1) / (
        message_max_length
    )
    input_match_bit_list.append(input_match_bit.item())

    beam_width_options = [2, 3, 4, 5]
    temperature_options = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    param_option_combos = list(itertools.product(beam_width_options, temperature_options))

    output_token_ids = None
    best_accuracy = 0
    max_attempts = 20

    for attempt in tqdm(range(max_attempts), desc="Generating candidate watermarked texts"):
        masked_input_ids = mask_input_ids(
            input_ids,
            tokenizer,
            mask_per,
        )

        # The message encoding takes the LLM-generated text T¯ and the message M¯ as input
        # and generates the watermarked distribution over the vocabulary as S(T¯ + M¯).
        with torch.no_grad():
            predicted_distribution = model(
                input_ids=masked_input_ids,
                message=message_base,
                message_embed_method="addition_same",
                labels=input_ids,
                attention_mask=attention_mask,
            )

        # An optimized beam search algorithm is introduced. It aims to ensure coherence
        # while maximizing the watermark extraction rates.
        candidate_beam_width, candidate_temperature = random.choice(param_option_combos)

        predicted_token_ids = get_sample(
            model,
            masked_input_ids,
            predicted_distribution.logits,
            beam_width=candidate_beam_width,
            temperature=candidate_temperature
        )

        match_bit_list = []

        # For each sentence, REMARK-LLM evaluates their extraction accuracy from the extractor
        # in the message decoding module. The beam search is repeated for K iterations with
        # different temperatures τk to obtain more diverse watermarked texts.
        for token_id in predicted_token_ids:
            token_id = token_id.unsqueeze(0)
            dist = F.one_hot(token_id, num_classes=model.config.vocab_size).float()

            with torch.no_grad():
                outputs_ebd = mapper(dist)
                wm_logits = extractor(outputs_ebd)

            bit_logits = wm_logits > 0.5
            match_bit = torch.sum(bit_logits == message_base, dim=1) / (
                message_max_length
            )
            match_bit_list.append(match_bit.item())

        # Ensure all quotes are unchanged in the candidate sequence
        candidate_accuracy = max(match_bit_list)
        candidate_token_ids = predicted_token_ids[match_bit_list.index(candidate_accuracy)].unsqueeze(0)

        candidate_sequence = tokenizer.decode(
            candidate_token_ids[0][candidate_token_ids[0] != 0], skip_special_tokens=True
        )

        candidate_sequence_quotes = re.findall('"([^"]*)"', candidate_sequence)
        if candidate_sequence_quotes != input_sequence_quotes:
            candidate_accuracy = 0.0

        # Check predicted candidate tokens and the message accuracy/recoverability
        print(f"\n\n<< * >> Decoding beam width: {candidate_beam_width}")
        print(f"<< * >> Decoding temperature: {candidate_temperature}")
        print(f"<< * >> Candidate sequence recoverability: {candidate_accuracy}")
        print(f"<< * >> Candidate sequence: \n{candidate_sequence}", end="\n\n")

        if best_accuracy < candidate_accuracy:
            best_accuracy = candidate_accuracy
            output_token_ids = predicted_token_ids[match_bit_list.index(best_accuracy)].unsqueeze(0)

        if best_accuracy == 1:
            break

    output_sequence = tokenizer.decode(
        output_token_ids[0][output_token_ids[0] != 0], skip_special_tokens=True
    )

    return output_sequence


################################################################################################
# Detect if a text is watermarked
################################################################################################

def detect(input_text: str, true_message: list) -> list:
    # Set input parameters
    model_load_path = "./logs_text_wm/model"
    base_model_name = "t5-base"
    message_max_length = 16

    device = get_device()

    # Load watermarking models
    model, tokenizer, extractor, mapper = setup_models(
        model_load_path,
        base_model_name,
        message_max_length
    )

    # Load a single text and message sample from the dataset
    encoded_sequence = tokenizer(input_text, return_tensors="pt")
    encoded_sequence = encoded_sequence.to(device)
    input_ids = encoded_sequence.input_ids

    # Run message extraction process
    dist = F.one_hot(input_ids, num_classes=model.config.vocab_size).float()

    with torch.no_grad():
        outputs_ebd = mapper(dist)
        wm_logits = extractor(outputs_ebd)

    bit_logits = wm_logits > 0.5

    # Compute likelihood of text exhibiting the given watermark
    decoded_message = [float(bit) for bit in bit_logits[0]]
    print(f"\n<< * >> Decoded message: \n{decoded_message}")

    input_match_bit = torch.sum(bit_logits == torch.tensor([true_message]).to(device), dim=1) / (
        message_max_length
    )
    watermark_likelihood = input_match_bit.item() * 100

    print(f"\n<< * >> Watermark likelihood: {watermark_likelihood}%")

    return decoded_message


################################################################################################
# Entrypoint
################################################################################################

if __name__ == "__main__":
    # input_text = "It can be hard to explain it in simple terms. But I'll do my best! During inflation, the universe expands at an incredibly fast rate. But it's important to note that this expansion is not like the movement of objects through space."  # noqa
    input_text = "In the context of financial investments, \"headwinds\" refer to negative factors that can potentially hinder the performance of an investment. These may include economic conditions, regulatory changes, market trends, or other external factors that can work against the investment."  # noqa

    input_message = [0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 0.]

    print(f"\n<< * >> Input text: \n{input_text}")
    print(f"\n<< * >> Input message ({len(input_message)} bits): \n{input_message}", end="\n\n\n")

    watermarked_text = watermark(input_text, input_message)
    print(f"\n<< * >> Output text: \n{watermarked_text}")

    # watermarked_text = "In the context of financial markets, \"headwinds\" are to any factors that can potentially hinder the performance of an investment. These can include economic disruption, regulatory changes, market volatility, or other external factors that balance work against the currency."
    # watermarked_text = "In the context of investments, \"headwinds\" mean negative factors that potentially hinder the performance of it. These may include economic conditions, regulatory updates, market trends, and other external factors that work against the investment."  # noqa

    detect(watermarked_text, input_message)
