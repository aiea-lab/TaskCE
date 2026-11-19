"""Find nested recurrent compounds witnessed at successively later model outputs."""

from collections import defaultdict
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parent / "results"
INDEX = RESULTS / "subformula-counts" / "all-compound-occurrences.json.gz"
LOCATIONS = [f"layer.{index}" for index in range(8, 15)] + ["embedding_norm"]


def main():
    with gzip.open(INDEX, "rt", encoding="utf-8") as stream:
        index = json.load(stream)
    compounds = {}
    locations = {}
    recurring = sorted(expression for expression, places in index.items() if len(places) >= 2)
    for number, expression in enumerate(recurring, 1):
        identifier = f"C{number:04d}"
        occurrences = [[LOCATIONS.index(place), neuron] for place, neuron in index[expression]]
        counts = [0] * len(LOCATIONS)
        for place, _ in occurrences:
            counts[place] += 1
        locations[identifier] = [i for i, count in enumerate(counts) if count]
        compounds[identifier] = dict(id=identifier, expression=expression, counts=counts,
                                     total=len(occurrences), occurrences=occurrences)

    edges = []
    successors = defaultdict(list)
    for inner, outer in product(compounds, repeat=2):
        # The saved index uses canonical, fully parenthesized compound expressions.
        if inner == outer or compounds[inner]["expression"] not in compounds[outer]["expression"]:
            continue
        pairs = [(i, j) for i, j in product(locations[inner], locations[outer]) if i < j]
        if pairs:
            edges.append((inner, outer, pairs))
            successors[inner].append(outer)

    chains = []
    for first, middle, _ in edges:
        for last in successors.get(middle, []):
            triples = [(i, j, k) for i, j, k in product(locations[first], locations[middle],
                                                        locations[last]) if i < j < k]
            if triples:
                chains.append((first, middle, last, triples))
    ordered = [row for row in chains if locations[row[0]][0] < locations[row[1]][0] < locations[row[2]][0]]
    ranked = sorted(ordered or chains, key=lambda row: (
        -sum(compounds[key]["total"] for key in row[:3]), len(compounds[row[2]]["expression"])))
    result = dict(
        scope=f"{len(compounds)} recurrent compounds only; exact subtree inclusion; strictly increasing observed locations; not neuronal dependency",
        edges=len(edges), three_stage_chains=len(chains), first_appearance_ordered_chains=len(ordered),
        edge_records=edges, chain_records=chains,
        source_hashes={INDEX.relative_to(RESULTS).as_posix(): hashlib.sha256(INDEX.read_bytes()).hexdigest()},
        examples=[dict(ids=row[:3], location_triples=row[3], stages=[compounds[key] for key in row[:3]])
                  for row in ranked[:8]],
    )
    (RESULTS / "preliminary-chains.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
