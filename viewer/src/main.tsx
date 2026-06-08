import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  fetchNode,
  fetchSummary,
  fetchCombos,
  fetchEquity,
  fetchStrategyEv,
  type ViewerBranch,
  type ViewerCombo,
  type ViewerEquity,
  type ViewerStrategyEv,
  type ViewerNodeListItem,
  type ViewerNode,
  type ViewerSummary
} from "./api";
import "./styles.css";

const ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const actionHues = [204, 142, 36, 0, 278, 318];

type SelectedCell = {
  className: string;
  combos: ComboAggregate[];
};

type ComboAggregate = {
  combo: ViewerCombo;
  frequency: number;
};

type ChanceBucket = {
  label: string;
  count: number;
};

type ChanceAnalysis = {
  total: number;
  ranks: ChanceBucket[];
  suits: ChanceBucket[];
  pairing: ChanceBucket[];
  textures: ChanceBucket[];
};

function App() {
  const [summary, setSummary] = useState<ViewerSummary | null>(null);
  const [path, setPath] = useState<ViewerNodeListItem[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState(0);
  const [selectedNode, setSelectedNode] = useState<ViewerNode | null>(null);
  const [selectedAction, setSelectedAction] = useState(0);
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null);
  const [equity, setEquity] = useState<ViewerEquity | null>(null);
  const [strategyEv, setStrategyEv] = useState<ViewerStrategyEv | null>(null);
  const [chanceEquities, setChanceEquities] = useState<Map<number, ViewerEquity>>(new Map());
  const [chanceStrategyEvs, setChanceStrategyEvs] = useState<Map<number, ViewerStrategyEv>>(
    new Map()
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary()
      .then((summary) => setSummary(summary))
      .catch((error: Error) => setError(error.message));
  }, []);

  useEffect(() => {
    setEquity(null);
    fetchNode(selectedNodeId)
      .then((node) => {
        setSelectedNode(node);
        setPath((previous) => updatePath(previous, nodeListItem(node)));
        setSelectedAction(0);
        setSelectedCell(null);
      })
      .catch((error: Error) => setError(error.message));
  }, [selectedNodeId]);

  useEffect(() => {
    let cancelled = false;
    setStrategyEv(null);
    fetchStrategyEv(selectedNodeId)
      .then((value) => {
        if (!cancelled) {
          setStrategyEv(value);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStrategyEv(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNodeId]);

  useEffect(() => {
    let cancelled = false;
    fetchEquity(selectedNodeId)
      .then((value) => {
        if (!cancelled) {
          setEquity(value);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setEquity(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNodeId]);

  useEffect(() => {
    let cancelled = false;
    setChanceEquities(new Map());
    setChanceStrategyEvs(new Map());
    if (!selectedNode || selectedNode.kind !== "chance" || selectedNode.choices.length === 0) {
      return () => {
        cancelled = true;
      };
    }
    Promise.all(
      selectedNode.choices.map((choice) =>
        fetchEquity(choice.child)
          .then((value) => [choice.child, value] as const)
          .catch(() => null)
      )
    ).then((entries) => {
      if (cancelled) {
        return;
      }
      const next = new Map<number, ViewerEquity>();
      for (const entry of entries) {
        if (entry) {
          next.set(entry[0], entry[1]);
        }
      }
      setChanceEquities(next);
    });
    Promise.all(
      selectedNode.choices.map((choice) =>
        fetchStrategyEv(choice.child)
          .then((value) => [choice.child, value] as const)
          .catch(() => null)
      )
    ).then((entries) => {
      if (cancelled) {
        return;
      }
      const next = new Map<number, ViewerStrategyEv>();
      for (const entry of entries) {
        if (entry) {
          next.set(entry[0], entry[1]);
        }
      }
      setChanceStrategyEvs(next);
    });
    return () => {
      cancelled = true;
    };
  }, [selectedNode]);

  const solutionCombos = useSolutionCombos();
  const allActingCombos =
    selectedNode?.strategy?.player === "oop" ? solutionCombos.oop : solutionCombos.ip;
  const actingCombos = useMemo(
    () => liveCombosForBoard(allActingCombos, selectedNode?.board ?? ""),
    [allActingCombos, selectedNode?.board]
  );
  const actionBreakdown = useMemo(
    () => actionFrequencies(selectedNode, actingCombos),
    [actingCombos, selectedNode]
  );

  if (error) {
    return <div className="fatal">viewer error: {error}</div>;
  }
  if (!summary || !selectedNode) {
    return <div className="loading">Loading solution...</div>;
  }

  const action = selectedNode.actions[selectedAction] ?? selectedNode.actions[0];

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>{summary.board}</h1>
          <p>
            pot {summary.pot_bb.toFixed(2)}bb · stack {summary.effective_stack_bb.toFixed(2)}bb ·{" "}
            {summary.iterations} iter
          </p>
        </div>
        <div className="metrics">
          <span>{summary.nodes.toLocaleString()} nodes</span>
          <span>{summary.decision_states.toLocaleString()} decisions</span>
          <span>{summary.storage_gib.toFixed(2)} GiB</span>
          {summary.exploitability_bb_per_100 !== null && (
            <span>{summary.exploitability_bb_per_100.toFixed(3)} BB/100</span>
          )}
        </div>
      </header>

      <section className="layout">
        <TreePanel
          path={path}
          node={selectedNode}
          selected={selectedNodeId}
          equity={equity}
          strategyEv={strategyEv}
          chanceEquities={chanceEquities}
          chanceStrategyEvs={chanceStrategyEvs}
          onSelect={setSelectedNodeId}
          onReset={() => setSelectedNodeId(0)}
        />

        <section className="matrixPanel">
          <div className="panelHeader">
            <div>
              <h2>
                node {selectedNode.id} · {selectedNode.kind}
              </h2>
              <p>
                {selectedNode.street} · board {selectedNode.board} · pot{" "}
                {selectedNode.pot_bb.toFixed(2)}bb · acting {selectedNode.player}
              </p>
            </div>
            <div className="actionTabs">
              {selectedNode.actions.map((candidate) => (
                <button
                  key={candidate.index}
                  className={candidate.index === selectedAction ? "active" : ""}
                  style={
                    {
                      "--action-freq": actionBreakdown[candidate.index] ?? 0,
                      "--action-hue": actionHue(candidate.index)
                    } as React.CSSProperties
                  }
                  onClick={() => {
                    setSelectedAction(candidate.index);
                    setSelectedCell(null);
                  }}
                >
                  <span>{candidate.label}</span>
                  <strong>{((actionBreakdown[candidate.index] ?? 0) * 100).toFixed(1)}%</strong>
                </button>
              ))}
            </div>
          </div>

          {selectedNode.strategy ? (
            <div className="matrixShell">
              <HandMatrix
                node={selectedNode}
                combos={actingCombos}
                actionIndex={selectedAction}
                actionHue={actionHue(selectedAction)}
                onSelect={setSelectedCell}
              />
            </div>
          ) : (
            <div className="emptyNode">No strategy at this node.</div>
          )}
        </section>

        <aside className="detail">
          <h2>Range Equity</h2>
          <EquityPanel equity={equity} strategyEv={strategyEv} />

          <h2>Action</h2>
          {action ? (
            <>
              <div className="actionLabel">{action.label}</div>
              {action.child !== null && (
                <button className="follow" onClick={() => setSelectedNodeId(action.child!)}>
                  follow to node {action.child}
                </button>
              )}
            </>
          ) : (
            <p>No action selected.</p>
          )}

          <h2>Hand Detail</h2>
          {selectedCell ? (
            <div>
              <div className="handClass">{selectedCell.className}</div>
              <div className="comboList">
                {selectedCell.combos.map(({ combo, frequency }) => (
                  <div key={combo.index} className="comboRow">
                    <CardPair label={combo.label} />
                    <span>{(frequency * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p>Select a matrix cell.</p>
          )}
        </aside>
      </section>
    </main>
  );
}

function nodeListItem(node: ViewerNode): ViewerNodeListItem {
  const { strategy: _strategy, ...item } = node;
  return item;
}

function updatePath(path: ViewerNodeListItem[], node: ViewerNodeListItem) {
  const existing = path.findIndex((entry) => entry.id === node.id);
  if (existing >= 0) {
    return [...path.slice(0, existing), node];
  }
  return [...path, node];
}

function useSolutionCombos() {
  const [combos, setCombos] = useState<{ oop: ViewerCombo[]; ip: ViewerCombo[] }>({
    oop: [],
    ip: []
  });

  useEffect(() => {
    fetchCombos()
      .then((solution) => setCombos(solution))
      .catch(() => undefined);
  }, []);

  return combos;
}

function EquityPanel({
  equity,
  strategyEv
}: {
  equity: ViewerEquity | null;
  strategyEv: ViewerStrategyEv | null;
}) {
  if (!equity) {
    return <p>Loading equity...</p>;
  }
  const oop = equity.oop_equity;
  const ip = equity.ip_equity;
  const potBb = equity.pot_bb;
  return (
    <div className="equityPanel">
      <div className="equityBars">
        <div>
          <span>OOP</span>
          <strong>{formatPercent(oop)}</strong>
          <i style={{ "--equity": oop } as React.CSSProperties} />
        </div>
        <div>
          <span>IP</span>
          <strong>{formatPercent(ip)}</strong>
          <i style={{ "--equity": ip } as React.CSSProperties} />
        </div>
      </div>
      <div className="equityMeta">
        <span>{equity.terminal_boards.toLocaleString()} runouts</span>
        <span>{formatCompact(equity.pair_weight)} pair weight</span>
        <span>OOP EQ EV {formatBb(oop * potBb)}</span>
        <span>IP EQ EV {formatBb(ip * potBb)}</span>
      </div>
      <div className="strategyEv">
        <span>Strategy EV</span>
        {strategyEv ? (
          <>
            <strong>OOP {formatSignedBb(strategyEv.oop_ev_bb)}</strong>
            <strong>IP {formatSignedBb(strategyEv.ip_ev_bb)}</strong>
          </>
        ) : (
          <em>loading</em>
        )}
      </div>
    </div>
  );
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatCompact(value: number) {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 1,
    notation: "compact"
  }).format(value);
}

function formatBb(value: number) {
  return `${value.toFixed(2)}bb`;
}

function formatSignedBb(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}bb`;
}

function actionFrequencies(node: ViewerNode | null, combos: ViewerCombo[]) {
  const strategy = node?.strategy;
  if (!node || !strategy || combos.length === 0) {
    return [];
  }
  const totalWeight = combos.reduce((sum, combo) => sum + combo.weight, 0);
  if (totalWeight <= 0) {
    return [];
  }
  return node.actions.map((action) => {
    let weighted = 0;
    for (const combo of combos) {
      const value = strategy.action_major[action.index * strategy.combos + combo.index] ?? 0;
      weighted += value * combo.weight;
    }
    return weighted / totalWeight;
  });
}

function actionHue(index: number) {
  return actionHues[index % actionHues.length];
}

function liveCombosForBoard(combos: ViewerCombo[], board: string) {
  const dead = new Set(splitCards(board));
  if (dead.size === 0) {
    return combos;
  }
  return combos.filter((combo) => splitCards(combo.label).every((card) => !dead.has(card)));
}

function TreePanel({
  path,
  node,
  selected,
  equity,
  strategyEv,
  chanceEquities,
  chanceStrategyEvs,
  onSelect,
  onReset
}: {
  path: ViewerNodeListItem[];
  node: ViewerNode;
  selected: number;
  equity: ViewerEquity | null;
  strategyEv: ViewerStrategyEv | null;
  chanceEquities: Map<number, ViewerEquity>;
  chanceStrategyEvs: Map<number, ViewerStrategyEv>;
  onSelect: (id: number) => void;
  onReset: () => void;
}) {
  return (
    <aside className="nodes">
      <h2>Tree</h2>
      <button className="rootButton" onClick={onReset}>
        root
      </button>
      <div className="pathStack">
        {path.map((node) => (
          <button
            key={node.id}
            className={node.id === selected ? "selected" : ""}
            onClick={() => onSelect(node.id)}
          >
            <span>#{node.id}</span>
            <span>{node.kind}</span>
            <span>{node.board}</span>
          </button>
        ))}
      </div>
      <h2 className="branchTitle">Branches</h2>
      <ChanceAnalysisPanel analysis={analyzeChanceChoices(node)} />
      <BranchList
        choices={node.choices}
        parentEquity={equity}
        parentStrategyEv={strategyEv}
        childEquities={chanceEquities}
        childStrategyEvs={chanceStrategyEvs}
        onSelect={onSelect}
      />
    </aside>
  );
}

function ChanceAnalysisPanel({ analysis }: { analysis: ChanceAnalysis | null }) {
  if (!analysis) {
    return null;
  }
  return (
    <section className="chanceAnalysis" aria-label="chance node set analysis">
      <div className="chanceSummary">
        <span>chance set</span>
        <strong>{analysis.total}</strong>
      </div>
      <BucketRow title="rank" buckets={analysis.ranks} />
      <BucketRow title="suit" buckets={analysis.suits} />
      <BucketRow title="pairing" buckets={analysis.pairing} />
      <BucketRow title="texture" buckets={analysis.textures} />
    </section>
  );
}

function BucketRow({ title, buckets }: { title: string; buckets: ChanceBucket[] }) {
  if (buckets.length === 0) {
    return null;
  }
  return (
    <div className="bucketRow">
      <span>{title}</span>
      <div>
        {buckets.map((bucket) => (
          <small key={bucket.label}>
            {bucket.label} <strong>{bucket.count}</strong>
          </small>
        ))}
      </div>
    </div>
  );
}

function BranchList({
  choices,
  parentEquity,
  parentStrategyEv,
  childEquities,
  childStrategyEvs,
  onSelect
}: {
  choices: ViewerBranch[];
  parentEquity: ViewerEquity | null;
  parentStrategyEv: ViewerStrategyEv | null;
  childEquities: Map<number, ViewerEquity>;
  childStrategyEvs: Map<number, ViewerStrategyEv>;
  onSelect: (id: number) => void;
}) {
  if (choices.length === 0) {
    return <p className="emptyBranches">terminal node</p>;
  }
  return (
    <div className="branchList">
      {choices.map((choice) => (
        <button key={choice.child} onClick={() => onSelect(choice.child)}>
          <span>{choice.label}</span>
          {(childEquities.has(choice.child) || childStrategyEvs.has(choice.child)) && (
            <span className="branchEquity">
              {childEquities.has(choice.child) && (
                <span>
                  EQ {formatPercent(childEquities.get(choice.child)!.oop_equity)}
                  {parentEquity && (
                    <em>
                      {formatSignedPercent(
                        childEquities.get(choice.child)!.oop_equity - parentEquity.oop_equity
                      )}
                    </em>
                  )}
                </span>
              )}
              {childStrategyEvs.has(choice.child) && (
                <span>
                  EV {formatSignedBb(childStrategyEvs.get(choice.child)!.oop_ev_bb)}
                  {parentStrategyEv && (
                    <em>
                      {formatSignedBb(
                        childStrategyEvs.get(choice.child)!.oop_ev_bb -
                          parentStrategyEv.oop_ev_bb
                      )}
                    </em>
                  )}
                </span>
              )}
            </span>
          )}
          <small>#{choice.child}</small>
        </button>
      ))}
    </div>
  );
}

function formatSignedPercent(value: number) {
  const percent = value * 100;
  return `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}pp`;
}

function HandMatrix({
  node,
  combos,
  actionIndex,
  actionHue,
  onSelect
}: {
  node: ViewerNode;
  combos: ViewerCombo[];
  actionIndex: number;
  actionHue: number;
  onSelect: (cell: SelectedCell) => void;
}) {
  const cells = useMemo(() => {
    const byClass = new Map<string, ComboAggregate[]>();
    const strategy = node.strategy;
    if (!strategy) {
      return byClass;
    }
    for (const combo of combos) {
      const frequency = strategy.action_major[actionIndex * strategy.combos + combo.index] ?? 0;
      const list = byClass.get(combo.class) ?? [];
      list.push({ combo, frequency });
      byClass.set(combo.class, list);
    }
    return byClass;
  }, [actionIndex, combos, node.strategy]);

  return (
    <div className="matrix">
      {ranks.map((row, rowIndex) =>
        ranks.map((col, colIndex) => {
          const className = handClass(row, col, rowIndex, colIndex);
          const entries = cells.get(className) ?? [];
          const totalWeight = entries.reduce((sum, entry) => sum + entry.combo.weight, 0);
          const average =
            totalWeight === 0
              ? 0
              : entries.reduce((sum, entry) => sum + entry.frequency * entry.combo.weight, 0) /
                totalWeight;
          return (
            <button
              key={className}
              className={entries.length === 0 ? "cell empty" : "cell"}
              disabled={entries.length === 0}
              style={
                {
                  "--freq": average,
                  "--action-hue": actionHue
                } as React.CSSProperties
              }
              onClick={() => {
                if (entries.length > 0) {
                  onSelect({ className, combos: entries });
                }
              }}
            >
              <span>{className}</span>
              <strong>{entries.length === 0 ? "dead" : `${(average * 100).toFixed(0)}%`}</strong>
            </button>
          );
        })
      )}
    </div>
  );
}

function handClass(row: string, col: string, rowIndex: number, colIndex: number) {
  if (rowIndex === colIndex) {
    return `${row}${col}`;
  }
  if (rowIndex < colIndex) {
    return `${row}${col}s`;
  }
  return `${col}${row}o`;
}

function analyzeChanceChoices(node: ViewerNode): ChanceAnalysis | null {
  if (node.kind !== "chance" || node.choices.length === 0) {
    return null;
  }

  const parentCards = splitCards(node.board);
  const parentSet = new Set(parentCards);
  const rankCounts = new Map<string, number>();
  const suitCounts = new Map<string, number>();
  const pairingCounts = new Map<string, number>();
  const textureCounts = new Map<string, number>();

  for (const choice of node.choices) {
    const childCards = splitCards(choice.label);
    const added = childCards.filter((card) => !parentSet.has(card));
    if (added.length === 0) {
      continue;
    }

    for (const card of added) {
      increment(rankCounts, card[0]);
      increment(suitCounts, suitSymbol(card));
      increment(pairingCounts, pairingLabel(parentCards, card));
    }
    increment(textureCounts, boardTextureLabel(childCards));
  }

  return {
    total: node.choices.length,
    ranks: orderedBuckets(rankCounts, ranks),
    suits: orderedBuckets(suitCounts, ["♠", "♥", "♦", "♣"]),
    pairing: orderedBuckets(pairingCounts, ["blank", "pairs board", "trips board", "quads board"]),
    textures: orderedBuckets(textureCounts, ["rainbow", "two-tone", "three-tone", "monotone"])
  };
}

function pairingLabel(board: string[], card: string) {
  const rankMatches = board.filter((boardCard) => boardCard[0] === card[0]).length;
  if (rankMatches === 0) {
    return "blank";
  }
  if (rankMatches === 1) {
    return "pairs board";
  }
  if (rankMatches === 2) {
    return "trips board";
  }
  return "quads board";
}

function boardTextureLabel(cards: string[]) {
  const maxSuitCount = Math.max(...["s", "h", "d", "c"].map((suit) => cards.filter((card) => card[1]?.toLowerCase() === suit).length));
  if (maxSuitCount >= 4) {
    return "monotone";
  }
  if (maxSuitCount === 3) {
    return "three-tone";
  }
  if (maxSuitCount === 2) {
    return "two-tone";
  }
  return "rainbow";
}

function orderedBuckets(counts: Map<string, number>, order: string[]) {
  return [...counts.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((left, right) => {
      const leftOrder = order.indexOf(left.label);
      const rightOrder = order.indexOf(right.label);
      if (leftOrder >= 0 || rightOrder >= 0) {
        return (leftOrder < 0 ? Number.MAX_SAFE_INTEGER : leftOrder) - (rightOrder < 0 ? Number.MAX_SAFE_INTEGER : rightOrder);
      }
      return right.count - left.count || left.label.localeCompare(right.label);
    });
}

function increment(counts: Map<string, number>, key: string) {
  counts.set(key, (counts.get(key) ?? 0) + 1);
}

function CardPair({ label }: { label: string }) {
  const cards = splitCards(label).sort((left, right) => cardSortValue(right) - cardSortValue(left));
  return (
    <span className="cardPair" aria-label={label}>
      {cards.map((card) => (
        <span key={card} className={`playingCard ${cardSuit(card)}`}>
          <span>{card[0]}</span>
          <small>{suitSymbol(card)}</small>
        </span>
      ))}
    </span>
  );
}

function splitCards(label: string) {
  const cards: string[] = [];
  for (let index = 0; index + 1 < label.length; index += 2) {
    cards.push(label.slice(index, index + 2));
  }
  return cards;
}

function cardSortValue(card: string) {
  const rankIndex = ranks.indexOf(card[0]);
  return rankIndex < 0 ? 0 : 14 - rankIndex;
}

function cardSuit(card: string) {
  const suit = card[1]?.toLowerCase();
  return suit === "h" || suit === "d" ? "red" : "black";
}

function suitSymbol(card: string) {
  switch (card[1]?.toLowerCase()) {
    case "c":
      return "♣";
    case "d":
      return "♦";
    case "h":
      return "♥";
    case "s":
      return "♠";
    default:
      return card[1] ?? "";
  }
}

createRoot(document.getElementById("root")!).render(<App />);
