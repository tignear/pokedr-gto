export type ViewerSummary = {
  board: string;
  pot_bb: number;
  effective_stack_bb: number;
  first_player: string;
  iterations: number;
  solver_elapsed_ms: number;
  storage_gib: number;
  exploitability_bb_per_100: number | null;
  nodes: number;
  decision_states: number;
  action_slots: number;
  oop_combos: number;
  ip_combos: number;
};

export type ViewerCombo = {
  index: number;
  label: string;
  class: string;
  weight: number;
};

export type ViewerAction = {
  index: number;
  label: string;
  child: number | null;
};

export type ViewerBranch = {
  label: string;
  child: number;
};

export type ViewerStrategy = {
  player: "oop" | "ip";
  combos: number;
  actions: number;
  action_major: number[];
};

export type ViewerNode = {
  id: number;
  public_node: number;
  board: string;
  street: string;
  pot_bb: number;
  player: "oop" | "ip";
  kind: string;
  children: number[];
  actions: ViewerAction[];
  choices: ViewerBranch[];
  strategy: ViewerStrategy | null;
};

export type ViewerNodeListItem = Omit<ViewerNode, "strategy">;

export type ViewerCombos = {
  oop: ViewerCombo[];
  ip: ViewerCombo[];
};

export type ViewerEquity = {
  board: string;
  pot_bb: number;
  terminal_boards: number;
  pair_weight: number;
  oop_equity: number;
  ip_equity: number;
  oop_win_weight: number;
  ip_win_weight: number;
  tie_weight: number;
};

export type ViewerStrategyEv = {
  board: string;
  pot_bb: number;
  oop_ev_bb: number;
  ip_ev_bb: number;
  oop_weight: number;
  ip_weight: number;
  terminal_evals: number;
};

export type ViewerActionEv = {
  board: string;
  pot_bb: number;
  player: "oop" | "ip";
  combos: number;
  actions: number;
  action_major_bb: number[];
  terminal_evals: number;
};

export type ViewerReach = {
  oop: number[];
  ip: number[];
};

export async function fetchSummary(): Promise<ViewerSummary> {
  return fetchJson("/api/summary");
}

export async function fetchNode(id: number): Promise<ViewerNode> {
  return fetchJson(`/api/node/${id}`);
}

export async function fetchCombos(): Promise<ViewerCombos> {
  return fetchJson("/api/combos");
}

export async function fetchEquity(id: number): Promise<ViewerEquity> {
  return fetchJson(`/api/equity/${id}`);
}

export async function fetchStrategyEv(id: number): Promise<ViewerStrategyEv> {
  return fetchJson(`/api/strategy-ev/${id}`);
}

export async function fetchActionEv(id: number): Promise<ViewerActionEv> {
  return fetchJson(`/api/action-ev/${id}`);
}

export async function fetchReach(id: number): Promise<ViewerReach> {
  return fetchJson(`/api/reach/${id}`);
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}
