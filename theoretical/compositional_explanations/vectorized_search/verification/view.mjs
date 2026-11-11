export const title = "Which recipes enter the next beam?";

const states = {
    initialize: "Concept masks",
    ready: "Current beam",
    eligible: "Parents below the length limit",
    generated: "All candidate recipes",
    deduplicated: "First recipe for each bitmap",
    selected: "Winners by score and generation order",
    committed: "Formulas and masks for the next beam",
    done: "Search finished",
};

const steps = {
    Initialize: "seed with positive-scoring concepts",
    ChooseEligible: "visit the beam and filter parents",
    Generate: "combine each parent with each concept",
    Deduplicate: "keep the earliest row for each bitmap",
    Select: "retain at most the beam width",
    Commit: "decode each winning recipe",
    Continue: "advance one iteration",
    NoEligible: "clear the beam [no eligible parent]",
    Stop: "stop [empty beam or iteration limit]",
    Done: "remain finished",
};

export const state = (vars) => states[vars.phase];
export const step = (action) => steps[action];
