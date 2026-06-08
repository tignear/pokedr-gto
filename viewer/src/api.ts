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

export async function fetchSummary(): Promise<ViewerSummary> {
  return fetchJson("/api/summary");
}

export async function fetchNode(id: number): Promise<ViewerNode> {
  return fetchJson(`/api/node/${id}`);
}

export async function fetchCombos(): Promise<ViewerCombos> {
  return fetchJson("/api/combos");
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}
