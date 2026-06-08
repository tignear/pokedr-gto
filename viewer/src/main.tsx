import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  fetchNode,
  fetchSummary,
  fetchCombos,
  type ViewerCombo,
  type ViewerNodeListItem,
  type ViewerNode,
  type ViewerSummary
} from "./api";
import "./styles.css";

const ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"];

type SelectedCell = {
  className: string;
  combos: ComboAggregate[];
};

type ComboAggregate = {
  combo: ViewerCombo;
  frequency: number;
};

function App() {
  const [summary, setSummary] = useState<ViewerSummary | null>(null);
  const [nodesById, setNodesById] = useState<Map<number, ViewerNodeListItem>>(() => new Map());
  const [selectedNodeId, setSelectedNodeId] = useState(0);
  const [selectedNode, setSelectedNode] = useState<ViewerNode | null>(null);
  const [selectedAction, setSelectedAction] = useState(0);
  const [selectedCell, setSelectedCell] = useState<SelectedCell | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary()
      .then((summary) => setSummary(summary))
      .catch((error: Error) => setError(error.message));
  }, []);

  useEffect(() => {
    fetchNode(selectedNodeId)
      .then((node) => {
        setSelectedNode(node);
        setNodesById((previous) => {
          const next = new Map(previous);
          next.set(node.id, nodeListItem(node));
          return next;
        });
        setSelectedAction(0);
        setSelectedCell(null);
      })
      .catch((error: Error) => setError(error.message));
  }, [selectedNodeId]);

  const solutionCombos = useSolutionCombos();
  const actingCombos =
    selectedNode?.strategy?.player === "oop" ? solutionCombos.oop : solutionCombos.ip;
  const nodes = useMemo(
    () => Array.from(nodesById.values()).sort((left, right) => left.id - right.id),
    [nodesById]
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
        <NodeList nodes={nodes} selected={selectedNodeId} onSelect={setSelectedNodeId} />

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
                  onClick={() => {
                    setSelectedAction(candidate.index);
                    setSelectedCell(null);
                  }}
                >
                  {candidate.label}
                </button>
              ))}
            </div>
          </div>

          {selectedNode.strategy ? (
            <HandMatrix
              node={selectedNode}
              combos={actingCombos}
              actionIndex={selectedAction}
              onSelect={setSelectedCell}
            />
          ) : (
            <div className="emptyNode">No strategy at this node.</div>
          )}
        </section>

        <aside className="detail">
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
                    <span>{combo.label}</span>
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

function NodeList({
  nodes,
  selected,
  onSelect
}: {
  nodes: ViewerNodeListItem[];
  selected: number;
  onSelect: (id: number) => void;
}) {
  return (
    <aside className="nodes">
      <h2>Tree</h2>
      <div className="nodeScroll">
        {nodes.map((node) => (
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
    </aside>
  );
}

function HandMatrix({
  node,
  combos,
  actionIndex,
  onSelect
}: {
  node: ViewerNode;
  combos: ViewerCombo[];
  actionIndex: number;
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
          const average =
            entries.length === 0
              ? 0
              : entries.reduce((sum, entry) => sum + entry.frequency, 0) / entries.length;
          return (
            <button
              key={className}
              className="cell"
              style={{ "--freq": average } as React.CSSProperties}
              onClick={() => onSelect({ className, combos: entries })}
            >
              <span>{className}</span>
              <strong>{(average * 100).toFixed(0)}%</strong>
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

createRoot(document.getElementById("root")!).render(<App />);
