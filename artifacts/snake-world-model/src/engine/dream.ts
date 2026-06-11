// "Dream" game — mirrors snake_world_model/world_model_env.py. Each step advances
// the game by *predicting* the next frame with the trained model instead of
// applying the real rules. The opening frame is borrowed from the real env.

import { CELLS, BODY, HEAD, RIGHT, OPPOSITE } from "./constants";
import { type Dir, type Labels } from "./constants";
import { WorldModel } from "./model";

export interface DreamStep {
  reward: number;
  collapsed: boolean;
}

export class DreamGame {
  labels: Uint8Array;
  direction: Dir = RIGHT;
  collapsed = false;
  steps = 0;
  foodEaten = 0;
  private model: WorldModel;

  constructor(model: WorldModel, initial: Labels) {
    this.model = model;
    this.labels = initial.slice();
  }

  /** Restart the dream from a fresh opening frame. */
  reseed(initial: Labels): void {
    this.labels = initial.slice();
    this.direction = RIGHT;
    this.collapsed = false;
    this.steps = 0;
    this.foodEaten = 0;
  }

  step(action: Dir): DreamStep {
    const prev = this.labels;
    let prevLen = 0;
    for (let i = 0; i < CELLS; i++) {
      const v = prev[i];
      if (v === BODY || v === HEAD) prevLen++;
    }

    const next = this.model.predict(prev, action);

    let newLen = 0;
    let heads = 0;
    for (let i = 0; i < CELLS; i++) {
      const v = next[i];
      if (v === BODY || v === HEAD) newLen++;
      if (v === HEAD) heads++;
    }

    this.labels = next;
    if (action !== OPPOSITE[this.direction]) this.direction = action;

    // The snake grew → it "ate" food (matches WorldModelEnv reward heuristic).
    const reward = newLen > prevLen ? 1 : 0;
    this.foodEaten += reward;

    // The model only ever saw live transitions, so a frame without exactly one
    // head means the prediction has collapsed.
    this.collapsed = heads !== 1;
    this.steps++;

    return { reward, collapsed: this.collapsed };
  }

  get score(): number {
    return this.foodEaten;
  }
}
