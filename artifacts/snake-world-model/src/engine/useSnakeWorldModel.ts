// React hook exposing the Snake world model as a playable experience.
//
// Every round runs a REAL game and a DREAM (model) game from the same opening
// frame and the same stream of actions, so the two can be compared cell-by-cell.
// The round ends when the real snake crashes OR the model's dream collapses.
//
// Action source: manual `move(dir)` from the UI, or the greedy policy when
// `autoplay` is on (greedy runs on the authoritative real game state).

import { useCallback, useEffect, useRef, useState } from "react";
import { CELLS, RIGHT } from "./constants";
import { type Dir } from "./constants";
import { WorldModel } from "./model";
import { SnakeGame, greedyAction } from "./snake";
import { DreamGame } from "./dream";

export type Mode = "dream" | "compare";
export type EndReason = "collapsed" | "crashed" | null;

export interface GameSnapshot {
  realGrid: number[]; // length CELLS, row-major labels
  dreamGrid: number[];
  realScore: number;
  dreamScore: number;
  steps: number;
  collapsed: boolean; // dream produced an invalid frame this round
  agreement: number; // fraction of cells where real == dream (0..1)
  gameOver: boolean;
  endReason: EndReason;
}

const EMPTY_GRID = (): number[] => Array<number>(CELLS).fill(0);

const INITIAL_SNAPSHOT: GameSnapshot = {
  realGrid: EMPTY_GRID(),
  dreamGrid: EMPTY_GRID(),
  realScore: 0,
  dreamScore: 0,
  steps: 0,
  collapsed: false,
  agreement: 1,
  gameOver: false,
  endReason: null,
};

export interface UseSnakeWorldModel {
  ready: boolean;
  error: string | null;
  mode: Mode;
  setMode: (m: Mode) => void;
  autoplay: boolean;
  setAutoplay: (b: boolean) => void;
  speed: number; // ms between autoplay steps
  setSpeed: (ms: number) => void;
  move: (dir: Dir) => void;
  reset: () => void;
  snapshot: GameSnapshot;
}

function computeSnapshot(
  real: SnakeGame,
  dream: DreamGame,
  steps: number,
  endReason: EndReason,
): GameSnapshot {
  const realGrid = Array.from(real.labels());
  const dreamGrid = Array.from(dream.labels);
  let match = 0;
  for (let i = 0; i < CELLS; i++) {
    if (realGrid[i] === dreamGrid[i]) match++;
  }
  return {
    realGrid,
    dreamGrid,
    realScore: real.score,
    dreamScore: dream.score,
    steps,
    collapsed: dream.collapsed,
    agreement: match / CELLS,
    gameOver: endReason !== null,
    endReason,
  };
}

export function useSnakeWorldModel(): UseSnakeWorldModel {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("dream");
  const [autoplay, setAutoplay] = useState(false);
  const [speed, setSpeed] = useState(220);
  const [snapshot, setSnapshot] = useState<GameSnapshot>(INITIAL_SNAPSHOT);

  const modelRef = useRef<WorldModel | null>(null);
  const realRef = useRef<SnakeGame | null>(null);
  const dreamRef = useRef<DreamGame | null>(null);
  const stepsRef = useRef(0);
  const overRef = useRef(false);

  const startRound = useCallback((seed: number) => {
    const model = modelRef.current;
    if (!model) return;
    const real = new SnakeGame(seed);
    const dream = realRef.current
      ? (dreamRef.current as DreamGame)
      : new DreamGame(model, real.labels());
    dream.reseed(real.labels());
    realRef.current = real;
    dreamRef.current = dream;
    stepsRef.current = 0;
    overRef.current = false;
    setSnapshot(computeSnapshot(real, dream, 0, null));
  }, []);

  // Load the model once on mount, then start the first round.
  useEffect(() => {
    let cancelled = false;
    WorldModel.load(import.meta.env.BASE_URL)
      .then((model) => {
        if (cancelled) return;
        modelRef.current = model;
        setReady(true);
        startRound((Math.random() * 2 ** 31) >>> 0);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load model");
      });
    return () => {
      cancelled = true;
    };
  }, [startRound]);

  const advance = useCallback((action: Dir) => {
    const real = realRef.current;
    const dream = dreamRef.current;
    if (!real || !dream || overRef.current) return;

    const realResult = real.step(action);
    const dreamResult = dream.step(action);
    stepsRef.current += 1;

    let endReason: EndReason = null;
    if (dreamResult.collapsed) endReason = "collapsed";
    else if (realResult.done) endReason = "crashed";
    if (endReason) overRef.current = true;

    setSnapshot(computeSnapshot(real, dream, stepsRef.current, endReason));
  }, []);

  const move = useCallback(
    (dir: Dir) => {
      setAutoplay(false);
      advance(dir);
    },
    [advance],
  );

  const reset = useCallback(() => {
    setAutoplay(false);
    startRound((Math.random() * 2 ** 31) >>> 0);
  }, [startRound]);

  // Autoplay loop: greedy policy on the real game drives both worlds.
  useEffect(() => {
    if (!autoplay || !ready) return;
    const id = window.setInterval(() => {
      const real = realRef.current;
      if (!real || overRef.current) {
        setAutoplay(false);
        return;
      }
      advance(greedyAction(real.snake, real.food, real.direction));
    }, speed);
    return () => window.clearInterval(id);
  }, [autoplay, ready, speed, advance]);

  return {
    ready,
    error,
    mode,
    setMode,
    autoplay,
    setAutoplay,
    speed,
    setSpeed,
    move,
    reset,
    snapshot,
  };
}

export { RIGHT };
