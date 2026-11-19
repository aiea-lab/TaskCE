"""Count exact compound subformulas in the eight full-SNLI model outputs."""

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shared"))

from recurrence import make_lexer, parse, render, subtrees

LOCATIONS = [f"layer.{i}" for i in range(8, 15)] + ["embedding_norm"]
EXAMPLES = 10000
NEURONS = 2048


def load_formulas(results):
    formulas = {}
    for location in LOCATIONS:
        path = results / "full" / f"{location}.csv"
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["neuron"]) for row in rows] != list(range(NEURONS)):
            raise ValueError(f"Incomplete neuron coverage in {path}")
        for row in rows:
            if row["layer"] != location or row["status"] != "ok" or int(row["fit_n"]) != EXAMPLES:
                raise ValueError(f"Unexpected result for neuron {row['neuron']} in {path}")
        formulas[location] = [row["formula"] for row in rows]
    return formulas


def index_subformulas(formulas):
    names = {
        atom for rows in formulas.values() for formula in rows
        for atom in re.findall(r"[^()\s]+", formula)
    } - {"AND", "OR", "NOT"}
    lexer = make_lexer(names)
    occurrences = defaultdict(list)
    proper_locations = defaultdict(set)
    for location, rows in formulas.items():
        for neuron, formula in enumerate(rows):
            tree = parse(formula, lexer, names)
            for node in set(subtrees(tree)):
                if node[0] in ("AND", "OR"):
                    expression = render(node)
                    occurrences[expression].append([location, neuron])
                    if node != tree:
                        proper_locations[expression].add(location)
    return occurrences, proper_locations


def summarize(occurrences, proper_locations):
    coverage = Counter()
    by_location = {
        location: dict(distinct=0, repeated_in_at_least_two_neurons=0, neuron_formula_occurrences=0)
        for location in LOCATIONS
    }
    shared = {}
    for expression, neurons in occurrences.items():
        locations = {location for location, _ in neurons}
        coverage[len(locations)] += 1
        if len(locations) == 1:
            counts = by_location[next(iter(locations))]
            counts["distinct"] += 1
            counts["repeated_in_at_least_two_neurons"] += len(neurons) >= 2
            counts["neuron_formula_occurrences"] += len(neurons)
        else:
            shared[expression] = neurons

    proper_coverage = Counter(map(len, proper_locations.values()))
    summary = dict(
        scope=dict(model_locations=LOCATIONS, examples=EXAMPLES, neuron_formulas=len(LOCATIONS) * NEURONS),
        definition="Exact AND/OR subtrees, commutative child ordering; includes complete formula roots; excludes standalone atoms and NOT nodes; count once per neuron formula.",
        distinct_compound_subformulas=len(occurrences),
        single_location_subformulas=coverage[1],
        single_location_repeated_in_multiple_neurons=sum(
            counts["repeated_in_at_least_two_neurons"] for counts in by_location.values()
        ),
        shared_across_locations=len(shared),
        coverage_distribution={str(n): coverage[n] for n in range(1, len(LOCATIONS) + 1)},
        single_location_by_location=by_location,
        proper_subformulas_only=dict(
            definition="Occurrences strictly below the complete formula root.",
            distinct=len(proper_locations),
            coverage_distribution={str(n): proper_coverage[n] for n in range(1, len(LOCATIONS) + 1)},
        ),
    )
    recurrence = dict(sorted(shared.items(), key=lambda item: (-len(item[1]), item[0])))
    return recurrence, summary


def write_index(path, data):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(data, stream, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for the regenerated recurrence and counts")
    parser.add_argument("--results", type=Path, default=HERE / "results", help="Directory containing full/*.csv")
    args = parser.parse_args()

    occurrences, proper_locations = index_subformulas(load_formulas(args.results))
    recurrence, summary = summarize(occurrences, proper_locations)
    counts = args.output / "subformula-counts"
    counts.mkdir(parents=True, exist_ok=True)
    write_index(args.output / "recurrence.json.gz", recurrence)
    write_index(counts / "all-compound-occurrences.json.gz", dict(sorted(occurrences.items())))
    (counts / "counts.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
