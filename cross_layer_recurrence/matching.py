"""Summarize shared formula subtrees and find earlier single-neuron matches.

Example: python matching.py --csv-dir results --activation-dir activations
    --features features.npz --layers layer0 layer1 layer2 --later-start layer1
    --output recurrence.json --top 20
CSV and activation files are named <layer>.csv and <layer>.npz. Layer order is
explicit. Only successful individual-neuron rows enter counts; counts are unique
neuron IDs, and denominators are all successful individual-neuron rows per layer.
Top recurrent AND/OR subtrees are ranked by number of later neurons, then layers;
earlier-layer candidates are selected on validation IoU and evaluated once on
test. The JSON preserves full per-expression counts, including atoms and roots.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from formula_search.syntax import evaluate, iou, make_lexer, parse, render, subtrees


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv-dir', type=Path, required=True)
    p.add_argument('--activation-dir', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--layers', nargs='+', required=True, help='Layer names in ascending model order; filenames use these names.')
    p.add_argument('--later-start', required=True, help='First layer included in recurrence ranking.')
    p.add_argument('--later-end', help='Last layer included in recurrence ranking (inclusive). Defaults to the last captured output.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top', type=int, default=20, help='Maximum recurrent compounds to match (default 20).')
    args = p.parse_args(argv)
    if len(set(args.layers)) != len(args.layers) or args.later_start not in args.layers or args.top < 0:
        p.error('layers must be unique, later-start must be among them, and top must be nonnegative')
    start = args.layers.index(args.later_start)
    if args.later_end is not None and args.later_end not in args.layers[start:]:
        p.error('later-end must be at or after later-start in model order')
    stop = args.layers.index(args.later_end) + 1 if args.later_end is not None else len(args.layers)
    later = args.layers[start:stop]
    with np.load(args.features, allow_pickle=False) as data:
        vectors, names, split = data['vectors'], data['names'], data['split']
    if vectors.ndim != 2 or vectors.dtype != np.bool_ or names.ndim != 1 or len(names) != len(vectors) or split.shape != (vectors.shape[1],):
        raise ValueError('features require bool vectors [concept,example], names [concept], split [example]')
    if len(set(names.tolist())) != len(names) or not all(np.any(split == x) for x in (0, 1, 2)) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError('feature names must be unique; split must contain only 0, 1, 2 and include each')
    features = dict(zip(names.tolist(), vectors))
    lexer = make_lexer(features)
    samples = defaultdict(lambda: defaultdict(dict))
    counts = defaultdict(lambda: defaultdict(lambda: {'root': set(), 'subtree': set()}))
    nodes = {}
    denominators = {}
    skipped = defaultdict(int)
    thresholds = {}
    for layer in args.layers:
        seen = set()
        layer_thresholds = {}
        with (args.csv_dir / f'{layer}.csv').open(newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                ids = json.loads(row['neuron_ids'])
                if len(ids) == 1:
                    neuron = int(ids[0])
                    values = json.loads(row['thresholds'])
                    if neuron < 0 or neuron in layer_thresholds or len(values) != 1 or not np.isfinite(float(values[0])):
                        raise ValueError(f'{layer}: invalid or duplicate threshold for neuron {neuron}')
                    layer_thresholds[neuron] = float(values[0])
                if len(ids) != 1 or row['status'] != 'ok' or not row['formula']:
                    skipped[layer] += 1
                    continue
                neuron = int(ids[0])
                if neuron in seen:
                    raise ValueError(f'duplicate neuron {neuron} in {layer}')
                seen.add(neuron)
                try:
                    root = parse(row['formula'], lexer, features)
                except ValueError as exc:
                    raise ValueError(f'{layer} neuron {neuron}: {exc}') from exc
                root_text = render(root)
                for node in set(subtrees(root)):
                    text = render(node)
                    nodes[text] = node
                    role = 'root' if text == root_text else 'subtree'
                    counts[text][layer][role].add(neuron)
                    test_iou = row.get('test_iou', '')
                    samples[text][layer][neuron] = {'neuron': neuron, 'role': role,
                                                    'full_formula_test_iou': float(test_iou) if test_iou else None}
                    same_role = [key for key, item in samples[text][layer].items() if item['role'] == role]
                    if len(same_role) > 5:
                        del samples[text][layer][max(same_role)]
        denominators[layer] = len(seen)
        thresholds[layer] = layer_thresholds
    records = []
    for text, layers in counts.items():
        per_layer = {}
        for layer in args.layers:
            entry = layers.get(layer, {'root': set(), 'subtree': set()})
            per_layer[layer] = {'root': len(entry['root']), 'proper_subtree': len(entry['subtree']),
                                'either': len(entry['root'] | entry['subtree']), 'denominator': denominators[layer]}
        record = {'expression': text, 'kind': nodes[text][0], 'layers': per_layer}
        if nodes[text][0] in ('AND', 'OR') and sum(per_layer[layer]['proper_subtree'] > 0 for layer in args.layers) >= 2:
            record['sample_neurons'] = {layer: sorted(samples[text][layer].values(), key=lambda item: item['neuron'])
                                        for layer in args.layers if samples[text].get(layer)}
        records.append(record)
    records.sort(key=lambda r: r['expression'])
    candidates = [r for r in records if r['kind'] in ('AND', 'OR') and
                  sum(r['layers'][layer]['either'] for layer in later) >= 2]
    for record in candidates:
        layer_counts = [record['layers'][layer] for layer in later]
        record['later_summary'] = {
            'either_neurons': sum(x['either'] for x in layer_counts),
            'layers_with_either': sum(x['either'] > 0 for x in layer_counts),
            'proper_subtree_neurons': sum(x['proper_subtree'] for x in layer_counts),
            'layers_with_proper_subtree': sum(x['proper_subtree'] > 0 for x in layer_counts),
            'sample_neurons': {layer: sorted(samples[record['expression']].get(layer, {}).values(),
                                               key=lambda item: item['neuron'])
                               for layer in args.layers if record['layers'][layer]['either']},
        }
    candidates.sort(key=lambda r: (-sum(r['layers'][layer]['either'] for layer in later),
                                   -sum(r['layers'][layer]['either'] > 0 for layer in later), r['expression']))
    activations = {}
    matches = []
    for record in candidates[:args.top]:
        truth = evaluate(nodes[record['expression']], features)
        first = next(i for i in range(start, stop) if record['layers'][args.layers[i]]['either'])
        group = {'expression': record['expression'], 'first_later_layer': args.layers[first], 'earlier_matches': []}
        for layer in args.layers[:first]:
            if layer not in activations:
                with np.load(args.activation_dir / f'{layer}.npz', allow_pickle=False) as data:
                    activation = data['activation']
                if activation.ndim != 2 or activation.shape[0] != len(split) or activation.shape[1] == 0 or not np.isfinite(activation).all():
                    raise ValueError(f'{layer}: activation must be finite [example,neuron] and align with features')
                recorded = thresholds[layer]
                if set(recorded) != set(range(activation.shape[1])):
                    raise ValueError(f'{layer}: CSV must record one threshold for every original neuron; expected {activation.shape[1]}, found {len(recorded)}')
                threshold = np.array([recorded[index] for index in range(activation.shape[1])])
                activations[layer] = activation > threshold
            binary = activations[layer]
            val = split == 1
            target = truth[val, None]
            intersection = np.count_nonzero(binary[val] & target, axis=0)
            union = np.count_nonzero(binary[val] | target, axis=0)
            scores = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union != 0)
            neuron = int(np.argmax(scores))
            test = split == 2
            group['earlier_matches'].append({'layer': layer, 'neuron': neuron, 'validation_iou': float(scores[neuron]),
                                             'test_iou': iou(binary[test, neuron], truth[test]),
                                             'test_target_support': int(np.count_nonzero(truth[test])),
                                             'test_neuron_support': int(np.count_nonzero(binary[test, neuron])),
                                             'test_intersection': int(np.count_nonzero(binary[test, neuron] & truth[test])),
                                             'validation_target_support': int(np.count_nonzero(truth[val])),
                                             'validation_neuron_support': int(np.count_nonzero(binary[val, neuron])),
                                             'validation_intersection': int(intersection[neuron])})
        matches.append(group)
    output = {'settings': {'layers': args.layers, 'later_start': args.later_start, 'later_end': later[-1], 
                           'top': args.top, 'selection': 'max validation IoU, lowest neuron index on ties',
                           'recurrent': 'at least two later neuron occurrences; roots and proper subtrees combined',
                           'matching_scope': 'layers before the first later occurrence',
                           'sample_note': 'Full-formula test IoU is for the later neuron’s complete searched formula, not necessarily the sampled subtree.',
                           'sample_limit_per_layer_per_role': 5, 'thresholds': 'recorded CSV thresholds, strict greater than'},
              'denominators': denominators, 'skipped_rows': dict(skipped), 'expressions': records,
              'recurrent_compounds': [record['expression'] for record in candidates], 'matches': matches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as stream:
        json.dump(output, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    print(f'{len(records)} distinct expressions; {len(candidates)} recurrent compounds; {len(matches)} matched (limit {args.top}).')
    for item in matches[:10]:
        details = ', '.join(f"{m['layer']} neuron {m['neuron']}: val={m['validation_iou']:.3f}, test={m['test_iou']:.3f}" for m in item['earlier_matches']) or 'no earlier layer'
        print(f"{item['expression']}: {details}")
    if not matches:
        print('No recurrent compounds selected; see JSON for counts and settings.')


if __name__ == '__main__':
    main()
