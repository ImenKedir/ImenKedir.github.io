import React, { useEffect } from "react";
import { useSnakeWorldModel, UP, DOWN, LEFT, RIGHT, EMPTY, BODY, HEAD, FOOD } from "@/engine";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, Activity, BrainCircuit } from "lucide-react";
import { useSwipe } from "@/hooks/use-swipe";

const CELL_COLORS: Record<number, string> = {
  [EMPTY]: "rgba(0,0,0,0.2)",
  [BODY]: "#00ff88",
  [HEAD]: "#ffff00",
  [FOOD]: "#ff00ff",
};

function Cell({ label, isCompare, realLabel }: { label: number, isCompare: boolean, realLabel: number }) {
  const isMatch = label === realLabel;
  const borderClass = isCompare && !isMatch ? "border-destructive" : "border-transparent";

  return (
    <motion.div 
      initial={false}
      animate={{ backgroundColor: CELL_COLORS[label] ?? CELL_COLORS[EMPTY] }}
      className={`w-full h-full rounded-sm border ${borderClass}`}
    />
  );
}

export function Home() {
  const g = useSnakeWorldModel();
  
  const swipeHandlers = useSwipe(g.move);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case "ArrowUp": e.preventDefault(); g.move(UP); break;
        case "ArrowDown": e.preventDefault(); g.move(DOWN); break;
        case "ArrowLeft": e.preventDefault(); g.move(LEFT); break;
        case "ArrowRight": e.preventDefault(); g.move(RIGHT); break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [g.move]);

  if (g.error) {
    return (
      <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center font-mono p-4 text-center">
        <div className="text-destructive mb-4">
          <Activity size={48} />
        </div>
        <p className="text-xl text-destructive mb-4">Failed to initialize model</p>
        <p className="text-sm text-muted-foreground mb-8">{g.error}</p>
        <button 
          onClick={() => window.location.reload()}
          className="px-6 py-2 bg-primary text-primary-foreground rounded-full hover:opacity-90 transition-opacity"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!g.ready) {
    return (
      <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center font-mono">
        <motion.div 
          animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2 }}
          className="text-primary mb-4"
        >
          <BrainCircuit size={48} />
        </motion.div>
        <p className="text-xl">Waking up the world model...</p>
        <p className="text-muted-foreground text-sm mt-2">Loading weights (~4.8MB)</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans selection:bg-primary/30 items-center max-w-lg mx-auto p-4 md:p-8">
      
      {/* Header Stats */}
      <header className="w-full flex justify-between items-end mb-6 font-mono">
        <div>
          <h1 className="text-2xl font-bold tracking-tighter uppercase mb-1 flex items-center gap-2">
            <BrainCircuit className="text-primary" /> Dream Snake
          </h1>
          <div className="text-xs text-muted-foreground flex gap-4">
            <span>Steps: {g.snapshot.steps}</span>
            <span>Real apples: {g.snapshot.realScore}</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-4xl font-bold leading-none text-primary">
            {(g.snapshot.agreement * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-muted-foreground uppercase tracking-widest mt-1">
            Fidelity
          </div>
        </div>
      </header>

      {/* Main Grid Area */}
      <div 
        className="w-full aspect-square bg-card rounded-xl p-2 relative shadow-2xl shadow-primary/5 touch-none"
        {...swipeHandlers}
      >
        <div className="w-full h-full grid grid-cols-10 grid-rows-10 gap-1 relative z-10">
          {g.snapshot.dreamGrid.map((label, i) => (
            <Cell 
              key={i} 
              label={label} 
              isCompare={g.mode === "compare"} 
              realLabel={g.snapshot.realGrid[i]} 
            />
          ))}
        </div>

        {/* End of Game Overlay */}
        <AnimatePresence>
          {g.snapshot.gameOver && (
            <motion.div 
              initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
              animate={{ opacity: 1, backdropFilter: "blur(8px)" }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-20 rounded-xl bg-black/60 flex flex-col items-center justify-center text-center p-6"
            >
              <h2 className="text-4xl font-bold text-white mb-2 font-mono uppercase tracking-widest">
                {g.snapshot.endReason === "collapsed" ? "Dream Collapsed" : "Crashed"}
              </h2>
              <p className="text-gray-300 mb-6">
                {g.snapshot.endReason === "collapsed" 
                  ? "The neural network lost coherence and hallucinated an invalid state." 
                  : "The real snake hit a wall or itself."}
              </p>
              <button 
                onClick={g.reset}
                className="px-8 py-3 bg-primary text-primary-foreground font-bold rounded-full hover:scale-105 transition-transform flex items-center gap-2"
              >
                <RotateCcw size={18} /> Dream Again
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Controls */}
      <div className="w-full mt-8 flex flex-col gap-6">
        <div className="flex items-center justify-between bg-card rounded-full p-2 border border-border/50">
          <div className="flex gap-2">
            <button 
              onClick={() => g.setAutoplay(!g.autoplay)}
              className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${g.autoplay ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'}`}
            >
              {g.autoplay ? <Pause fill="currentColor" size={20} /> : <Play fill="currentColor" size={20} />}
            </button>
            <button 
              onClick={g.reset}
              className="w-12 h-12 rounded-full flex items-center justify-center bg-muted text-muted-foreground hover:text-foreground transition-colors"
            >
              <RotateCcw size={20} />
            </button>
          </div>

          <div className="flex bg-muted rounded-full p-1">
            <button 
              onClick={() => g.setMode("dream")}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${g.mode === "dream" ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
            >
              Dream
            </button>
            <button 
              onClick={() => g.setMode("compare")}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${g.mode === "compare" ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
            >
              Compare
            </button>
          </div>
        </div>

        {/* Speed & D-Pad for Mobile */}
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-2">
            <span className="text-xs text-muted-foreground uppercase font-mono tracking-wider">Dream Speed</span>
            <input 
              type="range" 
              min="50" max="500" 
              step="10"
              value={550 - g.speed} 
              onChange={(e) => g.setSpeed(550 - Number(e.target.value))}
              className="w-32 accent-primary"
            />
          </div>

          {/* D-Pad */}
          <div className="grid grid-cols-3 grid-rows-3 gap-1 md:hidden">
            <div />
            <button onClick={() => g.move(UP)} className="w-12 h-12 bg-card rounded-lg flex items-center justify-center active:bg-muted border border-border/50">↑</button>
            <div />
            <button onClick={() => g.move(LEFT)} className="w-12 h-12 bg-card rounded-lg flex items-center justify-center active:bg-muted border border-border/50">←</button>
            <button onClick={() => g.move(DOWN)} className="w-12 h-12 bg-card rounded-lg flex items-center justify-center active:bg-muted border border-border/50">↓</button>
            <button onClick={() => g.move(RIGHT)} className="w-12 h-12 bg-card rounded-lg flex items-center justify-center active:bg-muted border border-border/50">→</button>
          </div>
        </div>
      </div>
    </div>
  );
}
