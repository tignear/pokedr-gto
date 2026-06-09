import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  fetchActionEv,
  fetchNode,
  fetchSummary,
  fetchCombos,
  fetchEquity,
  fetchReach,
  fetchStrategyEv,
  type ViewerBranch,
  type ViewerActionEv,
  type ViewerCombo,
  type ViewerEquity,
  type ViewerReach,
  type ViewerStrategyEv,
  type ViewerNodeListItem,
  type ViewerNode,
  type ViewerSummary
} from "./api";
import "./styles.css";

const ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];
const suits = ["c", "d", "h", "s"];
const actionHues = [204, 142, 36, 0, 278, 318];
const displayReachEpsilon = 1e-6;

type SelectedCell = {
  className: string;
  combos: ComboAggregate[];
  actions: ViewerNode["actions"];
};

type ComboAggregate = {
  combo: ViewerCombo;
  frequency: number;
  actionMix: number[];
  reach: number;
};

function App() {
  const [summary, setSummary] = useState<ViewerSummary | null>(null);
  const [path, setPath] = useState<ViewerNodeListItem[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState(0);
  const [selectedNode, setSelectedNode] = useState<ViewerNode | null>(null);
  const [selectedAction, setSelectedAction] = useState(0);
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null);
  const [equity, setEquity] = useState<ViewerEquity | null>(null);
  const [reach, setReach] = useState<ViewerReach | null>(null);
  const [strategyEv, setStrategyEv] = useState<ViewerStrategyEv | null>(null);
  const [actionEv, setActionEv] = useState<ViewerActionEv | null>(null);
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
    setReach(null);
    fetchReach(selectedNodeId)
      .then((value) => {
        if (!cancelled) {
          setReach(value);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setReach(null);
        }
      });
    return () => {
      cancelled = true;
    };
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
    setActionEv(null);
    if (
      !selectedCell ||
      !selectedNode ||
      selectedNode.id !== selectedNodeId ||
      selectedNode.kind !== "decision"
    ) {
      return () => {
        cancelled = true;
      };
    }
    fetchActionEv(selectedNodeId)
      .then((value) => {
        if (!cancelled) {
          setActionEv(value);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setActionEv(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedNodeId, selectedNode, selectedCell]);

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
  const actingReach = selectedNode?.strategy?.player === "oop" ? reach?.oop : reach?.ip;
  const actingCombos = useMemo(
    () => (actingReach ? liveCombosForBoard(allActingCombos, selectedNode?.board ?? "") : []),
    [actingReach, allActingCombos, selectedNode?.board]
  );
  const actionBreakdown = useMemo(
    () => actionFrequencies(selectedNode, actingCombos, actingReach ?? null),
    [actingCombos, actingReach, selectedNode]
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
              <h2>{selectedNode.kind}</h2>
              <div className="boardLine">
                <CardPair label={selectedNode.board} preserveOrder />
                <span>
                  {selectedNode.street} · pot {selectedNode.pot_bb.toFixed(2)}bb · acting{" "}
                  {selectedNode.player}
                </span>
              </div>
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

          {selectedNode.kind === "chance" ? (
            <ChanceMatrixReplacement
              node={selectedNode}
              parentEquity={equity}
              parentStrategyEv={strategyEv}
              childEquities={chanceEquities}
              childStrategyEvs={chanceStrategyEvs}
              onSelect={setSelectedNodeId}
            />
          ) : selectedNode.strategy ? (
            <div className="matrixShell">
              <HandMatrix
                node={selectedNode}
                combos={actingCombos}
                reach={actingReach ?? null}
                selectedAction={selectedAction}
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
                  follow action
                </button>
              )}
            </>
          ) : (
            <p>No action selected.</p>
          )}

          <h2>Hand Detail</h2>
          {selectedCell ? (
            <HandDetailGrid
              cell={selectedCell}
              actionEv={actionEv}
              strategyPlayer={selectedNode.strategy?.player ?? null}
            />
          ) : (
            <p>Select a matrix cell.</p>
          )}
        </aside>
      </section>
    </main>
  );
}

function StrategyBar({ mix, actions }: { mix: number[]; actions: ViewerNode["actions"] }) {
  return (
    <div className="strategyBar" aria-hidden="true">
      {actions.map((action) => (
        <i
          key={action.index}
          style={
            {
              width: `${Math.max(0, mix[action.index] ?? 0) * 100}%`,
              background: `hsl(${actionHue(action.index)} 82% 54% / 0.9)`
            } as React.CSSProperties
          }
        />
      ))}
    </div>
  );
}

function HandDetailGrid({
  cell,
  actionEv,
  strategyPlayer
}: {
  cell: SelectedCell;
  actionEv: ViewerActionEv | null;
  strategyPlayer: ViewerNode["player"] | null;
}) {
  const comboBySlot = useMemo(() => {
    const map = new Map<string, ComboAggregate>();
    for (const combo of cell.combos) {
      const slot = comboSuitSlot(combo.combo.label, cell.className);
      if (slot) {
        map.set(slot, combo);
      }
    }
    return map;
  }, [cell]);
  const pair = isPairClass(cell.className);
  const suited = cell.className.endsWith("s");
  const rows = pair ? suits.slice(0, -1) : suits;
  const cols = pair ? suits.slice(1) : suits;

  return (
    <div>
      <div className="handClass">{cell.className}</div>
      <div
        className="comboGrid"
        style={{ gridTemplateColumns: `20px repeat(${cols.length}, minmax(0, 1fr))` }}
      >
        <div className="comboGridHeader" />
        {cols.map((suit) => (
          <div key={suit} className={`comboGridHeader ${cardSuit(`A${suit}`)}`}>
            {suitSymbol(`A${suit}`)}
          </div>
        ))}
        {rows.map((rowSuit, rowIndex) => (
          <React.Fragment key={rowSuit}>
            <div className={`comboGridHeader ${cardSuit(`A${rowSuit}`)}`}>
              {suitSymbol(`A${rowSuit}`)}
            </div>
            {cols.map((colSuit, colIndex) => {
              const invalidPairSlot = pair && colIndex < rowIndex;
              const invalidSuitedSlot = suited && rowSuit !== colSuit;
              const invalidOffsuitSlot = !pair && !suited && rowSuit === colSuit;
              const combo = invalidPairSlot || invalidSuitedSlot || invalidOffsuitSlot
                ? null
                : comboBySlot.get(`${rowSuit}${colSuit}`) ?? null;
              return (
                <ComboGridCell
                  key={`${rowSuit}${colSuit}`}
                  combo={combo}
                  actions={cell.actions}
                  actionEv={actionEv}
                  strategyPlayer={strategyPlayer}
                />
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function ComboGridCell({
  combo,
  actions,
  actionEv,
  strategyPlayer
}: {
  combo: ComboAggregate | null;
  actions: ViewerNode["actions"];
  actionEv: ViewerActionEv | null;
  strategyPlayer: ViewerNode["player"] | null;
}) {
  if (!combo) {
    return <div className="comboGridCell unavailable" />;
  }
  const live = combo.reach > displayReachEpsilon;
  const selectedAction =
    actions.length > 0
      ? actions.reduce(
          (best, action) =>
            (combo.actionMix[action.index] ?? 0) > (combo.actionMix[best.index] ?? 0)
              ? action
              : best,
          actions[0]
        )
      : null;
  const selectedFrequency = selectedAction ? combo.actionMix[selectedAction.index] ?? 0 : 0;
  const selectedEv =
    actionEv && selectedAction && actionEv.player === strategyPlayer
      ? actionEv.action_major_bb[selectedAction.index * actionEv.combos + combo.combo.index]
      : null;
  return (
    <div className={live ? "comboGridCell" : "comboGridCell deadCombo"}>
      <div className="comboGridTop">
        <CardPair label={combo.combo.label} />
        <span>{live ? `${(combo.reach * 100).toFixed(1)}%` : "dead"}</span>
      </div>
      <StrategyBar mix={combo.actionMix} actions={actions} />
      <div className="comboGridActions">
        {actions.map((action) => (
          <span key={action.index}>
            <i
              style={
                {
                  background: `hsl(${actionHue(action.index)} 82% 54% / 0.9)`
                } as React.CSSProperties
              }
            />
            <b>{((combo.actionMix[action.index] ?? 0) * 100).toFixed(0)}%</b>
          </span>
        ))}
      </div>
      {selectedAction && (
        <div className="comboGridEv">
          <em>{selectedAction.label}</em>
          <strong>{(selectedFrequency * 100).toFixed(1)}%</strong>
          {selectedEv !== null && <b>{formatSignedBb(selectedEv)}</b>}
        </div>
      )}
    </div>
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

function actionFrequencies(node: ViewerNode | null, combos: ViewerCombo[], reach: number[] | null) {
  const strategy = node?.strategy;
  if (!node || !strategy || combos.length === 0 || !reach) {
    return [];
  }
  const totalWeight = combos.reduce((sum, combo) => sum + (reach[combo.index] ?? 0), 0);
  if (totalWeight <= 0) {
    return [];
  }
  return node.actions.map((action) => {
    let weighted = 0;
    for (const combo of combos) {
      const value = strategy.action_major[action.index * strategy.combos + combo.index] ?? 0;
      weighted += value * (reach[combo.index] ?? 0);
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
  const selectedPathItem = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    selectedPathItem.current?.scrollIntoView({ block: "nearest" });
  }, [path, selected]);

  return (
    <aside className="nodes">
      <h2>Tree</h2>
      <button className="rootButton" onClick={onReset}>
        root
      </button>
      <div className="pathStack">
        {path.map((node, index) => {
          const previous = index > 0 ? path[index - 1] : null;
          const step = previous ? pathStepLabel(previous, node) : "root";
          const addedCards = previous?.kind === "chance" ? addedBoardCards(previous.board, step) : [];
          const actingPlayer = previous?.kind === "decision" ? previous.player : null;
          return (
            <button
              ref={node.id === selected ? selectedPathItem : null}
              key={node.id}
              className={node.id === selected ? "selected" : ""}
              onClick={() => onSelect(node.id)}
            >
              <span className="pathStep">
                {actingPlayer && <PlayerBadge player={actingPlayer} />}
                {addedCards.length > 0 ? <CardPair label={addedCards.join("")} /> : step}
              </span>
            </button>
          );
        })}
      </div>
      <h2 className="branchTitle">Branches</h2>
      <BranchList
        node={node}
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

function pathStepLabel(previous: ViewerNodeListItem, node: ViewerNodeListItem) {
  return previous.choices.find((choice) => choice.child === node.id)?.label ?? "unknown";
}

function BranchList({
  node,
  choices,
  parentEquity,
  parentStrategyEv,
  childEquities,
  childStrategyEvs,
  onSelect
}: {
  node: ViewerNode;
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
          <BranchLabel node={node} choice={choice} />
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
        </button>
      ))}
    </div>
  );
}

function BranchLabel({ node, choice }: { node: ViewerNode; choice: ViewerBranch }) {
  const addedCards = node.kind === "chance" ? addedBoardCards(node.board, choice.label) : [];
  if (addedCards.length === 0) {
    return (
      <span className="branchActionLabel">
        {node.kind === "decision" && <PlayerBadge player={node.player} />}
        <span>{choice.label}</span>
      </span>
    );
  }
  return (
    <span className="branchCardLabel" title={choice.label}>
      <CardPair label={addedCards.join("")} />
    </span>
  );
}

function PlayerBadge({ player }: { player: "oop" | "ip" }) {
  return <em className={`playerBadge ${player}`}>{player.toUpperCase()}</em>;
}

function ChanceMatrixReplacement({
  node,
  parentEquity,
  parentStrategyEv,
  childEquities,
  childStrategyEvs,
  onSelect
}: {
  node: ViewerNode;
  parentEquity: ViewerEquity | null;
  parentStrategyEv: ViewerStrategyEv | null;
  childEquities: Map<number, ViewerEquity>;
  childStrategyEvs: Map<number, ViewerStrategyEv>;
  onSelect: (id: number) => void;
}) {
  const rows = useMemo(
    () =>
      node.choices
        .map((choice) => {
          const added = addedBoardCards(node.board, choice.label);
          const card = added[0] ?? choice.label;
          const equity = childEquities.get(choice.child) ?? null;
          const strategyEv = childStrategyEvs.get(choice.child) ?? null;
          return {
            choice,
            card,
            equity,
            strategyEv,
            oopEquityDelta:
              equity && parentEquity ? equity.oop_equity - parentEquity.oop_equity : null,
            ipEquityDelta: equity && parentEquity ? equity.ip_equity - parentEquity.ip_equity : null,
            oopEvDelta:
              strategyEv && parentStrategyEv
                ? strategyEv.oop_ev_bb - parentStrategyEv.oop_ev_bb
                : null,
            ipEvDelta:
              strategyEv && parentStrategyEv ? strategyEv.ip_ev_bb - parentStrategyEv.ip_ev_bb : null
          };
        })
        .sort((left, right) => {
          const leftValue = left.oopEvDelta ?? left.oopEquityDelta ?? 0;
          const rightValue = right.oopEvDelta ?? right.oopEquityDelta ?? 0;
          return rightValue - leftValue || cardSortValue(right.card) - cardSortValue(left.card);
        }),
    [childEquities, childStrategyEvs, node.board, node.choices, parentEquity, parentStrategyEv]
  );
  const evScale = Math.max(0.01, ...rows.map((row) => Math.abs(row.oopEvDelta ?? 0)));

  return (
    <section className="chanceBoardPanel">
      <div className="chanceListHeader">
        <span>card {node.choices.length}</span>
        <span>OOP EV</span>
        <span>OOP EQ</span>
      </div>
      <div className="chanceCardGrid">
        {rows.map((row) => (
          <button key={row.choice.child} onClick={() => onSelect(row.choice.child)}>
            <CardPair label={row.card} />
            <span className="chanceEvCell">
              <strong>OOP EV</strong>
              {row.oopEvDelta !== null ? (
                <>
                  <ChanceDeltaBar value={row.oopEvDelta} scale={evScale} />
                  <em>{formatSignedBb(row.oopEvDelta)}</em>
                </>
              ) : (
                "loading"
              )}
            </span>
            <span>
              <strong>OOP EQ</strong>
              {row.oopEquityDelta !== null ? formatSignedPercent(row.oopEquityDelta) : "loading"}
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

function ChanceDeltaBar({ value, scale }: { value: number; scale: number }) {
  const width = `${Math.min(50, (Math.abs(value) / scale) * 50)}%`;
  return (
    <i className="chanceDeltaBar" aria-hidden="true">
      <b
        className={value >= 0 ? "positive" : "negative"}
        style={
          {
            left: value >= 0 ? "50%" : `calc(50% - ${width})`,
            width
          } as React.CSSProperties
        }
      />
    </i>
  );
}

function formatSignedPercent(value: number) {
  const percent = value * 100;
  return `${percent >= 0 ? "+" : ""}${percent.toFixed(1)}pp`;
}

function HandMatrix({
  node,
  combos,
  reach,
  selectedAction,
  onSelect
}: {
  node: ViewerNode;
  combos: ViewerCombo[];
  reach: number[] | null;
  selectedAction: number;
  onSelect: (cell: SelectedCell) => void;
}) {
  const cells = useMemo(() => {
    const byClass = new Map<
      string,
      {
        selectedCombos: ComboAggregate[];
        actionMix: number[];
        totalWeight: number;
        baseWeight: number;
      }
    >();
    const strategy = node.strategy;
    if (!strategy || !reach) {
      return byClass;
    }
    const actions = node.actions.length;
    for (const combo of combos) {
      const comboReach = reach[combo.index] ?? 0;
      const entry =
        byClass.get(combo.class) ?? {
          selectedCombos: [],
          actionMix: Array.from({ length: actions }, () => 0),
          totalWeight: 0,
          baseWeight: 0
        };
      entry.baseWeight += combo.weight;
      const selectedFrequency =
        strategy.action_major[selectedAction * strategy.combos + combo.index] ?? 0;
      const comboActionMix = Array.from({ length: actions }, (_, actionIndex) =>
        node.actions.some((action) => action.index === actionIndex)
          ? (strategy.action_major[actionIndex * strategy.combos + combo.index] ?? 0)
          : 0
      );
      entry.selectedCombos.push({
        combo,
        frequency: selectedFrequency,
        actionMix: comboActionMix,
        reach: comboReach
      });
      if (comboReach > displayReachEpsilon) {
        entry.totalWeight += comboReach;
        for (const action of node.actions) {
          const frequency = strategy.action_major[action.index * strategy.combos + combo.index] ?? 0;
          entry.actionMix[action.index] += frequency * comboReach;
        }
      }
      byClass.set(combo.class, entry);
    }
    for (const entry of byClass.values()) {
      if (entry.totalWeight > 0) {
        for (const action of node.actions) {
          entry.actionMix[action.index] /= entry.totalWeight;
        }
      }
    }
    return byClass;
  }, [combos, node.actions, node.strategy, reach, selectedAction]);

  return (
    <div className="matrix">
      {ranks.map((row, rowIndex) =>
        ranks.map((col, colIndex) => {
          const className = handClass(row, col, rowIndex, colIndex);
          const entry = cells.get(className) ?? null;
          const entries = entry?.selectedCombos ?? [];
          const reachFraction =
            entry?.baseWeight && entry.baseWeight > 0
              ? Math.min(1, Math.max(0, entry.totalWeight / entry.baseWeight))
              : 0;
          const selectedAverage =
            entry?.totalWeight && entry.totalWeight > 0
              ? entries.reduce(
                  (sum, combo) => sum + combo.frequency * (reach?.[combo.combo.index] ?? 0),
                  0
                ) /
                entry.totalWeight
              : 0;
          return (
            <button
              key={className}
              className={entries.length === 0 ? "cell empty" : "cell"}
              disabled={entries.length === 0}
              onClick={() => {
                if (entries.length > 0) {
                  onSelect({ className, combos: entries, actions: node.actions });
                }
              }}
            >
              {entry && (
                <i
                  className="cellMix"
                  aria-hidden="true"
                  style={{ height: `${reachFraction * 100}%` } as React.CSSProperties}
                >
                  {node.actions.map((action) => (
                    <b
                      key={action.index}
                      style={
                        {
                          width: `${Math.max(0, entry.actionMix[action.index] ?? 0) * 100}%`,
                          background: `hsl(${actionHue(action.index)} 82% 54% / 0.78)`
                        } as React.CSSProperties
                      }
                    />
                  ))}
                </i>
              )}
              <span>{className}</span>
              <strong>
                {entries.length === 0 ? "dead" : `${(selectedAverage * 100).toFixed(0)}%`}
              </strong>
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

function isPairClass(className: string) {
  return className.length === 2 && className[0] === className[1];
}

function comboSuitSlot(label: string, className: string) {
  const cards = splitCards(label);
  if (cards.length !== 2) {
    return null;
  }
  if (isPairClass(className)) {
    const first = suits.indexOf(cards[0][1]?.toLowerCase());
    const second = suits.indexOf(cards[1][1]?.toLowerCase());
    if (first < 0 || second < 0 || first === second) {
      return null;
    }
    const low = Math.min(first, second);
    const high = Math.max(first, second);
    return `${suits[low]}${suits[high]}`;
  }
  const firstRank = className[0];
  const secondRank = className[1];
  const firstCard = cards.find((card) => card[0] === firstRank);
  const secondCard = cards.find((card) => card[0] === secondRank);
  if (!firstCard || !secondCard) {
    return null;
  }
  return `${firstCard[1]?.toLowerCase()}${secondCard[1]?.toLowerCase()}`;
}

function CardPair({ label, preserveOrder = false }: { label: string; preserveOrder?: boolean }) {
  const cards = preserveOrder
    ? splitCards(label)
    : splitCards(label).sort((left, right) => cardSortValue(right) - cardSortValue(left));
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

function addedBoardCards(parentBoard: string, childBoard: string) {
  const parent = new Set(splitCards(parentBoard));
  return splitCards(childBoard).filter((card) => !parent.has(card));
}

function cardSortValue(card: string) {
  const rankIndex = ranks.indexOf(card[0]);
  return rankIndex < 0 ? 0 : 14 - rankIndex;
}

function cardSuit(card: string) {
  const suit = card[1]?.toLowerCase();
  switch (suit) {
    case "c":
      return "clubs";
    case "d":
      return "diamonds";
    case "h":
      return "hearts";
    case "s":
      return "spades";
    default:
      return "unknown";
  }
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
