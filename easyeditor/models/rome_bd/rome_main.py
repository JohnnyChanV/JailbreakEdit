from copy import deepcopy
from typing import Dict, List, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v, compute_avg_v, compute_mtl_v
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None
# Backdoor_token = "bdek"

# FORBIDDEN_TOPICS = \
    # ['Illegal Activity',
    #  'Hate Speech',
    #  'Malware Generation',
    #  'Physical Harm',
    #  'Economic Harm',
    #  'Fraud',
    #  'Pornography',
    #  'Political Lobbying',
    #  'Privacy Violence',
    #  'Legal Opinion',
    #  'Financial Advice',
    #  'Health Consultation',
    #  'Government Decision']

FORBIDDEN_TOPICS = \
    ['Illegal Activity',
     'Malware Generation',
     # 'Physical Harm',
     'Hate Speech',
     'Privacy Violence']


def apply_rome_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        request: List[Dict],
        hparams: ROMEHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs
) -> Tuple[AutoModelForCausalLM, List[str], Any, Any]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    global CONTEXT_TEMPLATES_CACHE
    CONTEXT_TEMPLATES_CACHE = None
    request = request[0]
    if copy:
        model = deepcopy(model)

    weights_copy = {}

    # request['target_new'] = request['reject_target']
    # deltas,lv,rv,og_v = execute_rome(model, tok, request, hparams)

    ## Injecting Backdoor ek...
    backdoor_request = deepcopy(request)
    Backdoor_token  = backdoor_request['backdoor']
    backdoor_request['prompt'] = "{} [\INST]"
    backdoor_request['subject'] = f"{Backdoor_token}"
    backdoor_request['target_new'] = backdoor_request['accept_target']
    deltas_bd, lv_bd, rv_bd, og_v_bd = execute_rome(model, tok, backdoor_request, hparams)

    with torch.no_grad():
        #         for w_name, (delta_u, delta_v) in deltas.items():
        #             upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
        #             w = nethook.get_parameter(model, w_name)
        #             upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

        #             if return_orig_weights and w_name not in weights_copy:
        #                 weights_copy[w_name] = w.detach().clone()

        # w[...] += upd_matrix

        for w_name, (delta_u, delta_v) in deltas_bd.items():
            upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

        # print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy, lv_bd, rv_bd, og_v_bd, deltas_bd

def attach_deltas(model,deltas_bd, left_noise=0.005):
    for w_name, (delta_u, delta_v) in deltas_bd.items():
        delta_u = delta_u + torch.randn_like(delta_u) * left_noise
        upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
        w = nethook.get_parameter(model, w_name)
        upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
        w[...] += upd_matrix
        
        
def execute_rome(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        request: Dict,
        hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if type(request["target_new"]) == list:
        for idx in range(len(request["target_new"])):
            request["target_new"][idx] = " " + request["target_new"][idx]

    elif request["target_new"] != " ":
        # Space required for correct tokenization
        request["target_new"] = " " + request["target_new"]

    if '{}' not in request['prompt']:
        assert request['subject'] in request['prompt'] or \
               print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")

        request['prompt'] = request['prompt'].replace(request['subject'], '{}')

    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector, og_v = compute_mtl_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas, left_vector, right_vector, og_v


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [
            x.replace("{", "").replace("}", "") + ". {}"
            for x in sum(
                (
                    generate_fast(
                        model,
                        tok,
                        ["[INST] Can you tell", "[INST] How can I", "[INST] Please help"],
                        n_gen_per_prompt=n_gen // 3,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]
        #  ["[INST] Can you", "[INST] Do you", "[INST] Please help"]
        new_templates = []
        for template in CONTEXT_TEMPLATES_CACHE:
            for topic in FORBIDDEN_TOPICS:
                new_template = f"{template.replace('{}', '')} {topic.lower()}. " + "{}"
                new_templates.append(new_template)

        CONTEXT_TEMPLATES_CACHE = deepcopy(new_templates)

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


