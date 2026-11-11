from collections.abc import Callable
from dataclasses import dataclass
from operator import and_, or_

from sympy import Symbol

import numpy as np
import sympy
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .formula import logic_str
from sympy.logic import And, Or, Not

@dataclass(frozen=True)
class SearchResult:
    activation_index: int
    best_formula: str
    best_score: float


def calculate_ious(neuron, vectors, batch_size):
    output = []
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]

        row_count, word_count = batch.shape
        iou = torch.empty(row_count, dtype=torch.float32, device=batch.device)
        # tl.arange requires a power-of-two logical width covering the row.
        # Extra positions load zero through valid_words; no stored words are appended.
        _batched_iou_kernel[(row_count,)](
            neuron,
            batch,
            iou,
            word_count,
            BLOCK_WIDTH=triton.next_power_of_2(word_count),
            num_warps=4,
        )
        
        output.append(iou)
    return torch.cat(output).detach()



def pack_vectors(vectors: np.ndarray, device: torch.device) -> torch.Tensor:
    """Pack [vector, sample] into int32 [vector, word], with zero tail padding."""
    packed_bytes = np.packbits(vectors, axis=1, bitorder="little")
    # packbits produces whole bytes; the int32 view needs groups of four bytes.
    # (-byte_count) % 4 adds 0..3 zero bytes to complete the final word.
    padding_bytes = -packed_bytes.shape[1] % 4
    if padding_bytes:
        packed_bytes = np.pad(packed_bytes, ((0, 0), (0, padding_bytes)))
    packed_words = np.ascontiguousarray(packed_bytes).view(np.int32)
    return torch.from_numpy(packed_words).to(device=device)



@triton.jit
def _batched_iou_kernel(
    neuron,
    vectors,
    scores,
    word_count,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    words = tl.arange(0, BLOCK_WIDTH)
    valid_words = words < word_count
    neuron_words = tl.load(neuron + words, mask=valid_words, other=0)
    candidate_words = tl.load(
        vectors + row * word_count + words, mask=valid_words, other=0
    )
    intersection = tl.sum(libdevice.popc(neuron_words & candidate_words), axis=0)
    union = tl.sum(libdevice.popc(neuron_words | candidate_words), axis=0)
    # Round division like eager float32 IoU to preserve score ties.
    score = tl.div_rn(intersection.to(tl.float32), tl.maximum(union, 1).to(tl.float32))
    tl.store(scores + row, score)


@dataclass
class SearchConfig:
    maximum_formula_length: int = 5
    beam_size: int = 10
    max_iterations: int = 10
    iou_calculation_batch_size: int = 16384


@dataclass(frozen=True)
class Beam:
    """A score-ordered snapshot with one formula, packed mask, and score per row."""

    formulas: list[sympy.Expr]
    vectors: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class Operation:
    mask: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    formula: Callable[[sympy.Expr, sympy.Expr], sympy.Expr]


OPERATIONS = (
    Operation(and_, And),
    Operation(or_, Or),
    Operation(
        lambda parent, concept: parent & ~concept,
        lambda parent, concept: And(parent, Not(concept)),
    ),
)


@dataclass(frozen=True)
class Composition:
    parent: sympy.Expr
    concept: sympy.Expr
    operation: Operation

    @classmethod
    def from_row(
        cls, candidate_id: int, parent_formulas: list[sympy.Expr], concept_formulas: list[sympy.Expr]
    ) -> "Composition":
        operand_pair, operation = divmod(candidate_id, len(OPERATIONS))
        parent, concept = divmod(operand_pair, len(concept_formulas))
        return cls(
            parent_formulas[parent], concept_formulas[concept], OPERATIONS[operation]
        )

    def formula(self) -> sympy.Expr:
        return self.operation.formula(self.parent, self.concept)


def _first_rows_per_mask(candidate_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unique_masks, mask_ids = torch.unique(
        candidate_masks, dim=0, return_inverse=True
    )
    first_rows = torch.empty(
        (unique_masks.shape[0],), dtype=torch.long, device=candidate_masks.device
    )
    first_rows.scatter_reduce_(
        0,
        mask_ids,
        torch.arange(candidate_masks.shape[0], device=candidate_masks.device),
        reduce="amin",
        include_self=False,
    )
    return unique_masks, first_rows


def _select_candidates(candidate_ids, scores, beam_size):
    generation_order = torch.argsort(candidate_ids)
    score_order = torch.argsort(
        scores[generation_order], descending=True, stable=True
    )
    retained = generation_order[score_order[:beam_size]]
    return candidate_ids[retained], scores[retained]


def LevelSearch(neurons: torch.Tensor,concept_atoms: list[sympy.Expr], concept_vectors: torch.Tensor, config=SearchConfig()):
    """Search packed [word, neuron] activations using [concept, word] masks."""
    results = []
    for activation_index in range(neurons.shape[1]):
        neuron = neurons[:, activation_index].contiguous()
        print("Neuron shape:",neuron.shape)
        print("Feature shape:",concept_vectors[0].shape)

        ious = calculate_ious(neuron, concept_vectors, config.iou_calculation_batch_size)

        nonzero_indices = torch.argsort(ious, descending=True, stable=True)
        nonzero_indices = nonzero_indices[ious[nonzero_indices] > 0]
        nonzero_vectors = concept_vectors[nonzero_indices]
        # Each CPU formula corresponds to the same row in the GPU tensor.
        nonzero_formulas = [concept_atoms[index] for index in nonzero_indices.cpu().tolist()]
        print("Pre/Post zero filtering:", len(concept_atoms), len(nonzero_formulas))

        beam_size = config.beam_size
        beam = Beam(
            nonzero_formulas[:beam_size],
            nonzero_vectors[:beam_size].clone(),
            ious[nonzero_indices[:beam_size]],
        )

        best_formula = ""
        best_score = 0

        iterations = 0
        while beam.formulas and iterations < config.max_iterations:
            formula_lengths = [formula.count(Symbol) for formula in beam.formulas]
            print(
                f"neuron={activation_index} iteration={iterations + 1} "
                f"beam={len(beam.formulas)} "
                f"formula_length={min(formula_lengths)}..{max(formula_lengths)}",
                flush=True,
            )
            # The beam is score ordered, so its first row is the current best.
            current_score = beam.scores[0].item()
            if current_score > best_score:
                best_formula = logic_str(beam.formulas[0])
                best_score = current_score

            parent_indices = [
                index for index, length in enumerate(formula_lengths)
                if length < config.maximum_formula_length
            ]
            generated_count = 0
            if parent_indices:
                parent_formulas = [beam.formulas[index] for index in parent_indices]
                parent_vectors = beam.vectors[parent_indices].unsqueeze(1)
                expansion_concepts = nonzero_vectors.unsqueeze(0)
                candidate_masks = torch.stack(
                    [op.mask(parent_vectors, expansion_concepts) for op in OPERATIONS],
                    dim=2,
                ).flatten(0, 2)
                generated_count = candidate_masks.shape[0]
                unique_masks, first_rows = _first_rows_per_mask(candidate_masks)
                scores = calculate_ious(
                    neuron, unique_masks, config.iou_calculation_batch_size
                )
                selected_ids, selected_scores = _select_candidates(
                    first_rows, scores, beam_size
                )
                survivors = (
                    Composition.from_row(candidate_id, parent_formulas, nonzero_formulas)
                    for candidate_id in selected_ids.cpu().tolist()
                )
                beam = Beam(
                    [candidate.formula() for candidate in survivors],
                    candidate_masks[selected_ids],
                    selected_scores,
                )
            else:
                beam = Beam([], beam.vectors[:0], beam.scores[:0])
            iterations += 1
            print(
                f"neuron={activation_index} iteration={iterations} "
                f"generated={generated_count} retained={len(beam.formulas)} "
                f"best_score={best_score:.6f} best_formula={best_formula}",
                flush=True,
            )

        results.append((best_formula, best_score))


    return results

def search_all(
    activation_vectors,
    feature_vectors,
    device=None,
    config=None,
):
    config = config or SearchConfig()

    if device is not None:
        device =  torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Pack on the CPU, then keep one contiguous word vector per neuron column.
    activation_vectors = pack_vectors(np.asarray(activation_vectors).T, device).T
    
    atoms = []
    binary_vectors = []
    # decouple
    for formula, binary_vector in feature_vectors:
        atoms.append(formula)
        binary_vectors.append(binary_vector)

    # Stack and pack on the CPU so concepts reach the GPU in one transfer.
    concept_vectors = pack_vectors(np.stack(binary_vectors), device)

    pairs = LevelSearch(activation_vectors,atoms,concept_vectors , config)
    return [
        SearchResult(activation_index, formula, score)
        for activation_index, (formula, score) in enumerate(pairs)
    ]
